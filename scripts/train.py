"""Stage-aware modeling: temporal validation, model comparison, stage A/B,
ablations, calibration, interpretability, fairness.

Methodological guards
---------------------
* Temporal split: train <= 2026-04-30, validate 2026-05-01..06-30.
* Hyperparameters tuned on an INNER temporal split inside the train window
  (<= 2026-03-31 vs April) - the reporting validation is never used for tuning.
* Stage predictions apply the structural mask discovered in EDA: a component
  that was not advised cannot be the failure stage (label crosstab = exact zeros).
* Evaluation cohort is never touched here.

Run:  .venv/Scripts/python scripts/train.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, confusion_matrix,
                             brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ART, FIG, MODELS, TRAIN_END, VALID_START, STAGES, COMPLETED_STAGE,
                    STAGE2ID, ID2STAGE, SEED)

warnings.filterwarnings("ignore")

NUM_COLS = [
    "age", "duration_min", "review_due_days", "distance_km", "village_pop",
    "facility_cho_count", "facility_asha_count", "consult_weekday", "consult_month",
    "f_stockout_rate_6m", "f_low_stock_share", "hist_episodes", "hist_rx_meds",
    "hist_disp_rate", "hist_partial_rate", "hist_stockout_rate", "hist_test_rate",
    "hist_review_rate", "hist_any_followup_rate", "hist_recent_dropout",
    "days_since_last_rx", "last_systolic_bp", "last_glucose", "last_bmi",
    "days_since_ncd_screen", "prior_visits", "prior_referrals", "prior_outreach",
    "prior_outreach_unreachable",
    # derived
    "n_advised", "ncd_count",
]
CAT_COLS = [
    "gender", "known_ncd_status", "vulnerability_group", "preferred_language",
    "road_access", "mobile_connectivity", "district", "chief_complaint",
    "diagnosis_group", "consult_mode", "connectivity_quality", "consult_status",
    "facility_type", "facility_network", "ncd_control_status", "link_conf",
]
BIN_COLS = ["medicine_advised", "test_advised", "review_advised", "chronic_flag",
            "dist_x_medicine"]

ABLN = {
    "A_demographics": ["age", "gender", "vulnerability_group", "known_ncd_status",
                       "district", "village_pop", "road_access", "mobile_connectivity",
                       "preferred_language"],
    "B_demog_consult": ["age", "gender", "vulnerability_group", "known_ncd_status",
                        "district", "village_pop", "road_access", "mobile_connectivity",
                        "preferred_language", "chief_complaint", "diagnosis_group",
                        "consult_mode", "connectivity_quality", "consult_status",
                        "duration_min", "medicine_advised", "test_advised",
                        "review_advised", "review_due_days", "distance_km",
                        "consult_weekday", "consult_month", "facility_type",
                        "facility_network", "facility_cho_count", "facility_asha_count",
                        "n_advised", "chronic_flag", "ncd_count", "dist_x_medicine"],
}


def _expand_sets():
    hist = [c for c in NUM_COLS if c.startswith(("hist_", "prior_")) or
            c in ("days_since_last_rx", "last_systolic_bp", "last_glucose", "last_bmi",
                  "days_since_ncd_screen", "ncd_control_status")]
    stock = ["f_stockout_rate_6m", "f_low_stock_share"]
    sets = dict(ABLN)
    sets["C_plus_history"] = sets["B_demog_consult"] + hist + ["ncd_control_status"]
    sets["D_all"] = sets["C_plus_history"] + stock
    return sets


def add_derived(F):
    F = F.copy()
    F["consult_date"] = pd.to_datetime(F["consult_date"])
    for c in CAT_COLS:
        F[c] = F[c].astype("string").fillna("__missing__")
    F["n_advised"] = (pd.to_numeric(F["medicine_advised"], errors="coerce").fillna(0)
                      + pd.to_numeric(F["test_advised"], errors="coerce").fillna(0)
                      + pd.to_numeric(F["review_advised"], errors="coerce").fillna(0))
    F["chronic_flag"] = (F["known_ncd_status"].astype(str).str.lower() != "none").astype(int)
    F["ncd_count"] = F["known_ncd_status"].astype(str).apply(
        lambda s: 0 if s.lower() == "none" else s.count("+") + 1)
    F["dist_x_medicine"] = pd.to_numeric(F["distance_km"], errors="coerce").fillna(0) * \
        pd.to_numeric(F["medicine_advised"], errors="coerce").fillna(0)
    return F


def encode(F, cols):
    X = pd.DataFrame(index=F.index)
    for c in cols:
        if c in CAT_COLS:
            d = pd.get_dummies(F[c].astype(str), prefix=c, dtype=float)
            X = pd.concat([X, d], axis=1)
        else:
            X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


def topk(y, p, frac):
    n = max(int(len(y) * frac), 1)
    idx = np.argsort(-p)[:n]
    tp = float(y[idx].sum())
    return tp / n, tp / max(float(y.sum()), 1)


def cls_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "brier": brier_score_loss(y, p),
        "precision_at_10": topk(y, p, 0.10)[0], "recall_at_10": topk(y, p, 0.10)[1],
        "precision_at_20": topk(y, p, 0.20)[0], "recall_at_20": topk(y, p, 0.20)[1],
    }


def pipe(name, params=None):
    params = params or {}
    if name == "logistic_regression":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=SEED))])
    if name == "random_forest":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(
                             n_estimators=500, min_samples_leaf=params.get("min_samples_leaf", 10),
                             max_features="sqrt", n_jobs=-1, random_state=SEED))])
    if name == "grad_boost":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", GradientBoostingClassifier(
                             n_estimators=300, learning_rate=0.05, max_depth=3,
                             subsample=0.9, random_state=SEED))])
    if name == "xgboost":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", xgb.XGBClassifier(
                             n_estimators=400, learning_rate=0.05,
                             max_depth=params.get("max_depth", 4), subsample=0.9,
                             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                             eval_metric="logloss", random_state=SEED, n_jobs=-1))])
    if name == "lightgbm":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", lgb.LGBMClassifier(
                             n_estimators=400, learning_rate=0.05,
                             num_leaves=params.get("num_leaves", 31), subsample=0.9,
                             colsample_bytree=0.8, min_child_samples=params.get("min_child_samples", 30),
                             random_state=SEED, n_jobs=-1, verbose=-1))])
    raise ValueError(name)


def stage_mask(advised_df, classes):
    """Structural mask: forbid stages whose component was not advised.
    Returns boolean array aligned to `classes` order given as stage-id list."""
    n = len(advised_df)
    M = np.ones((n, len(classes)), dtype=bool)
    for j, sid in enumerate(classes):
        stg = ID2STAGE[sid]
        if stg == "Medicine not collected":
            M[:, j] &= advised_df["medicine_advised"].values.astype(bool)
        elif stg == "Test not completed":
            M[:, j] &= advised_df["test_advised"].values.astype(bool)
        elif stg == "Review not attended":
            M[:, j] &= advised_df["review_advised"].values.astype(bool)
        elif stg == COMPLETED_STAGE:
            pass  # always allowed
    return M


def main():
    F = pd.read_csv(os.path.join(ART, "features.csv"))
    F = add_derived(F)

    dev = F[(F.cohort == "DEVELOPMENT") & F.lost_to_followup_label.notna()].copy()
    ev = F[F.cohort == "EVALUATION"].copy()
    dev["y"] = dev["lost_to_followup_label"].astype(int)
    dev = dev[dev.dropout_stage_label.isin(STAGES)]  # dev rows always have stage labels
    train = dev[dev.consult_date <= pd.Timestamp(TRAIN_END)]
    valid = dev[dev.consult_date >= pd.Timestamp(VALID_START)]
    # inner tuning split (still temporal; inside train window only)
    tune_tr = train[train.consult_date <= pd.Timestamp("2026-03-31")]
    tune_va = train[train.consult_date >= pd.Timestamp("2026-04-01")]
    print(f"[data] train={len(train)} valid={len(valid)} eval={len(ev)} "
          f"| inner tune: {len(tune_tr)}/{len(tune_va)}")

    results = []
    p_base = np.full(len(valid), train.y.mean())
    results.append({"model": "baseline_majority_rate", "features": "train prevalence",
                    **{k: round(v, 4) for k, v in cls_metrics(valid.y.values, p_base).items()}})

    sets = _expand_sets()
    full_cols = sets["D_all"]

    # ---------------- inner-split hyperparameter search (temporal) ------------
    grid = {
        "random_forest": [{"min_samples_leaf": v} for v in (5, 10, 20)],
        "xgboost": [{"max_depth": v} for v in (3, 4, 5)],
        "lightgbm": [{"num_leaves": a, "min_child_samples": b}
                     for a in (15, 31) for b in (20, 40)],
    }
    Xt_t, Xv_t = encode(tune_tr, full_cols), encode(tune_va, full_cols).reindex(
        columns=encode(tune_tr, full_cols).columns, fill_value=0.0)
    best_params = {}
    for name, glist in grid.items():
        best, bp = -1, None
        for g in glist:
            m = pipe(name, g)
            m.fit(Xt_t.values, tune_tr.y.values)
            ap = average_precision_score(tune_va.y.values, m.predict_proba(Xv_t.values)[:, 1])
            if ap > best:
                best, bp = ap, g
        best_params[name] = bp
        print(f"[tune] {name}: best {bp} (inner AP {best:.3f})")

    # ---------------- primary comparison on temporal validation ---------------
    Xtr = encode(train, full_cols)
    Xv = encode(valid, full_cols).reindex(columns=Xtr.columns, fill_value=0.0)
    fitted = {}
    for name in ["logistic_regression", "random_forest", "grad_boost", "xgboost", "lightgbm"]:
        mdl = pipe(name, best_params.get(name, {}))
        mdl.fit(Xtr.values, train.y.values)
        p = mdl.predict_proba(Xv.values)[:, 1]
        m = cls_metrics(valid.y.values, p)
        results.append({"model": name, "features": "D_all",
                        **{k: round(v, 4) for k, v in m.items()}})
        fitted[name] = (mdl, p)
        print(f"[model] {name:20s} roc={m['roc_auc']:.3f} pr={m['pr_auc']:.3f} "
              f"f1={m['f1']:.3f} p@10={m['precision_at_10']:.3f} r@10={m['recall_at_10']:.3f}")
    best_name = max(fitted, key=lambda k: fitted[k][1].tolist() and
                    average_precision_score(valid.y.values, fitted[k][1]))
    best_name = max(fitted, key=lambda k: average_precision_score(valid.y.values, fitted[k][1]))
    mdl_best, p_best = fitted[best_name]
    print(f"[choose] best by temporal PR-AUC: {best_name}")

    # structural rules artifact (EDA evidence for stage masking)
    ct = pd.crosstab(
        [train.medicine_advised, train.test_advised, train.review_advised],
        train.dropout_stage_label)
    ct.to_csv(os.path.join(ART, "stage_structural_rules.csv"))

    # ---------------- stage models -------------------------------------------
    stage_cols = [c for c in full_cols]
    st_train = train[train.y == 1]
    st_valid = valid[valid.y == 1]
    Xs_tr = encode(st_train, stage_cols)
    Xs_v = encode(st_valid, stage_cols).reindex(columns=Xs_tr.columns, fill_value=0.0)
    ys_tr = st_train["dropout_stage_label"].map(STAGE2ID)
    stage_clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("clf", lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05,
                                                     num_leaves=31, min_child_samples=25,
                                                     class_weight="balanced",
                                                     random_state=SEED, n_jobs=-1, verbose=-1))])
    stage_clf.fit(Xs_tr.values, ys_tr.values)
    classes = list(stage_clf.named_steps["clf"].classes_)
    probs_b = stage_clf.predict_proba(Xs_v.values)
    Mb = stage_mask(st_valid, classes)
    probs_b = np.where(Mb, probs_b, -np.inf)
    stg_b = np.array(classes)[probs_b.argmax(1)]
    y_true_b = st_valid["dropout_stage_label"].map(STAGE2ID).values
    macro_b = f1_score(y_true_b, stg_b, average="macro")
    macro_b_nomask = f1_score(y_true_b, stage_clf.predict(Xs_v.values), average="macro")
    print(f"[stage-B] macro-F1 dropouts: masked={macro_b:.3f} unmasked={macro_b_nomask:.3f}")

    # Approach A: direct multiclass on all episodes
    Xa_tr = encode(train, stage_cols)
    Xa_v = encode(valid, stage_cols).reindex(columns=Xa_tr.columns, fill_value=0.0)
    ya_tr = train["dropout_stage_label"].map(STAGE2ID)
    direct = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                                  num_leaves=31, min_child_samples=25,
                                                  class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1, verbose=-1))])
    direct.fit(Xa_tr.values, ya_tr.values)
    cls_a = list(direct.named_steps["clf"].classes_)
    probs_a = direct.predict_proba(Xa_v.values)
    Ma = stage_mask(valid, cls_a)
    probs_a_masked = np.where(Ma, probs_a, -np.inf)
    stg_a = np.array(cls_a)[probs_a_masked.argmax(1)]
    y_true_a = valid["dropout_stage_label"].map(STAGE2ID).values
    drop_mask = valid.y.eq(1).values
    macro_a_all = f1_score(y_true_a, stg_a, average="macro")
    macro_a_drop = f1_score(y_true_a[drop_mask], stg_a[drop_mask], average="macro")
    print(f"[stage-A] macro-F1 all={macro_a_all:.3f} dropouts={macro_a_drop:.3f}")

    pd.DataFrame([
        {"approach": "A_direct_multiclass", "macro_f1_all_episodes": round(macro_a_all, 4),
         "macro_f1_dropout_only": round(macro_a_drop, 4)},
        {"approach": "B_hierarchical", "macro_f1_all_episodes": np.nan,
         "macro_f1_dropout_only": round(macro_b, 4)},
        {"approach": "B_hierarchical_no_mask", "macro_f1_all_episodes": np.nan,
         "macro_f1_dropout_only": round(macro_b_nomask, 4)},
    ]).to_csv(os.path.join(ART, "stage_comparison.csv"), index=False)

    chosen_stage = "B_hierarchical" if macro_b >= macro_a_drop else "A_direct"
    cm = confusion_matrix(y_true_b, stg_b, labels=list(STAGE2ID.values()))
    pd.DataFrame(cm, index=[f"true_{s}" for s in STAGES],
                 columns=[f"pred_{s}" for s in STAGES]).to_csv(
        os.path.join(ART, "stage_confusion_matrix.csv"))

    # ---------------- ablations (all with best model) -------------------------
    abl_rows = []
    for sname in ["A_demographics", "B_demog_consult", "C_plus_history", "D_all"]:
        cols = sets[sname]
        Xa_ = encode(train, cols)
        Xb_ = encode(valid, cols).reindex(columns=Xa_.columns, fill_value=0.0)
        m_ = pipe(best_name, best_params.get(best_name, {}))
        m_.fit(Xa_.values, train.y.values)
        p_ = m_.predict_proba(Xb_.values)[:, 1]
        mm = cls_metrics(valid.y.values, p_)
        abl_rows.append({"ablation": sname, "n_features": Xa_.shape[1],
                         **{k: round(v, 4) for k, v in mm.items()}})
        print(f"[ablation] {sname:16s} pr={mm['pr_auc']:.3f} roc={mm['roc_auc']:.3f} f1={mm['f1']:.3f}")
    # D minus stage-history
    hist_cols = [c for c in sets["D_all"] if c.startswith(("hist_", "prior_"))]
    cols = [c for c in sets["D_all"] if c not in hist_cols]
    Xa_ = encode(train, cols)
    Xb_ = encode(valid, cols).reindex(columns=Xa_.columns, fill_value=0.0)
    m_ = pipe(best_name, best_params.get(best_name, {}))
    m_.fit(Xa_.values, train.y.values)
    mm = cls_metrics(valid.y.values, m_.predict_proba(Xb_.values)[:, 1])
    abl_rows.append({"ablation": "D_all_minus_stage_history", "n_features": Xa_.shape[1],
                     **{k: round(v, 4) for k, v in mm.items()}})
    pd.DataFrame(abl_rows).to_csv(os.path.join(ART, "ablation.csv"), index=False)

    # ---------------- figures ------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        _, p = fitted[name]
        fp, mp = calibration_curve(valid.y.values, p, n_bins=8)
        ax[0].plot(mp, fp, marker="o", label=name)
        fpr, tpr, _ = roc_curve(valid.y.values, p)
        ax[1].plot(fpr, tpr, label=f"{name} ({roc_auc_score(valid.y.values, p):.3f})")
    prec, rec, _ = precision_recall_curve(valid.y.values, p_best)
    ax[1].plot(rec, prec, "k--", label=f"{best_name} PR ({average_precision_score(valid.y.values, p_best):.3f})")
    ax[0].plot([0, 1], [0, 1], ":", c="gray")
    ax[0].set_xlabel("mean predicted"); ax[0].set_ylabel("observed fraction")
    ax[0].set_title("Calibration - temporal validation"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("FPR  /  Recall"); ax[1].set_ylabel("TPR  /  Precision")
    ax[1].set_title("ROC & PR - temporal validation"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "calibration_roc_pr.png"), dpi=150)
    plt.close(fig)

    # ---------------- interpretability ---------------------------------------
    if best_name in ("xgboost", "lightgbm", "random_forest"):
        try:
            expl = shap.TreeExplainer(mdl_best.named_steps["clf"])
            Xs = mdl_best.named_steps["imp"].transform(Xv.values)
            sv = expl.shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[-1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, -1]
            plt.figure(figsize=(8, 6))
            shap.summary_plot(sv, Xv, show=False, max_display=15)
            plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_summary.png"), dpi=150)
            plt.close()
            imp_s = pd.Series(np.abs(sv).mean(0), index=Xv.columns).sort_values(ascending=False)
            imp_s.head(25).to_csv(os.path.join(ART, "shap_top_features.csv"),
                                  header=["mean_abs_shap"])
            print("[interpretability] SHAP top-10:", list(imp_s.head(10).index))
        except Exception as e:
            print(f"[interpretability] SHAP unavailable for {best_name}: {e}; "
                  "permutation importance below")
    perm = permutation_importance(mdl_best, Xv.values, valid.y.values, n_repeats=8,
                                  random_state=SEED, scoring="average_precision", n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=Xv.columns).sort_values(ascending=False)
    perm_imp.head(25).to_csv(os.path.join(ART, "permutation_importance.csv"),
                             header=["mean_perm_importance"])

    # ---------------- fairness / subgroups -----------------------------------
    vtmp = valid.copy(); vtmp["p"] = p_best
    vtmp["age_band"] = pd.cut(vtmp["age"], [0, 30, 45, 60, 200],
                              labels=["<=30", "31-45", "46-60", "60+"])
    fr = []
    for col in ["gender", "vulnerability_group", "age_band", "district"]:
        for g, gdf in vtmp.groupby(col, observed=True):
            if len(gdf) < 30 or gdf.y.nunique() < 2:
                continue
            pred = (gdf.p >= 0.5).astype(int)
            fr.append({"group_col": str(col), "group": str(g), "n": int(len(gdf)),
                       "prevalence": round(gdf.y.mean(), 3),
                       "precision": round(precision_score(gdf.y, pred, zero_division=0), 3),
                       "recall": round(recall_score(gdf.y, pred, zero_division=0), 3),
                       "fnr": round(1 - recall_score(gdf.y, pred, zero_division=0), 3)})
    pd.DataFrame(fr).to_csv(os.path.join(ART, "fairness.csv"), index=False)

    # ---------------- persist ------------------------------------------------
    import pickle
    with open(os.path.join(MODELS, "dropout_model.pkl"), "wb") as f:
        pickle.dump({"model_name": best_name, "pipeline": mdl_best,
                     "columns": list(Xtr.columns), "stage_cols": stage_cols,
                     "hyperparams": best_params}, f)
    with open(os.path.join(MODELS, "stage_model.pkl"), "wb") as f:
        pickle.dump({"pipeline": stage_clf, "columns": list(Xs_tr.columns),
                     "classes": classes}, f)

    res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    res.to_csv(os.path.join(ART, "model_comparison.csv"), index=False)
    json.dump({"chosen_model": best_name,
               "chosen_pr_auc": float(average_precision_score(valid.y.values, p_best)),
               "chosen_roc_auc": float(roc_auc_score(valid.y.values, p_best)),
               "stage_approach": chosen_stage,
               "macro_f1_B_masked": float(macro_b), "macro_f1_B_nomask": float(macro_b_nomask),
               "macro_f1_A_dropout": float(macro_a_drop),
               "hyperparams": best_params},
              open(os.path.join(ART, "chosen.json"), "w"), indent=1)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
