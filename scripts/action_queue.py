"""Priority engine, stage-specific actions, and final submission files.

Priority (transparent, capacity-calibrated on the temporal validation set):
    priority_score = 0.60*risk + 0.25*stage_urgency + 0.10*consequence + 0.05*feasibility
Tiering by validation-set score quantiles at configurable capacity shares
(P1 top 15%, P2 next 15%, P3 next 20%), then P4.

System-barrier distinction: high recent facility stock-out + medicine failure
flags a SYSTEM barrier -> escalate to facility (not patient non-adherence).

Evaluation cohort is scored with the model trained on ALL development data
(refit), stages via hierarchical model + structural mask, actions by stage.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ART, MODELS, SUB, DEV_END, VALID_START, EVAL_START, STAGES,
                    COMPLETED_STAGE, STAGE_SHORT, STAGE2ID, ID2STAGE, SEED)
from features import load_all, build_episodes, attach_linkage, build_reference_info, \
    canonicalize, prepare_stock, build_features
from train import encode

# ---------------------------------------------------------------- thresholds
CAPACITY = {"P1": 0.15, "P2": 0.15, "P3": 0.20, "P4": 0.50}   # configurable
WEIGHTS = {"risk": 0.60, "urgency": 0.25, "consequence": 0.10, "feasibility": 0.05}
STAGE_URGENCY = {"Medicine not collected": 1.0, "Test not completed": 0.75,
                 "Review not attended": 0.6, COMPLETED_STAGE: 0.0}

ACTIONS = {
    "Medicine not collected": (
        "Contact patient within 48h; confirm medicine availability at facility; "
        "facilitate pickup/home delivery; if stock-out confirmed, escalate to "
        "facility in-charge (system barrier, not patient non-adherence)"),
    "Test not completed": (
        "Call with test details and where to go; assist scheduling; arrange "
        "transport guidance; re-check completion after 7 days"),
    "Review not attended": (
        "Reminder before review due date; help book review slot; home visit if "
        "unreachable; follow up within 3 days of missed review"),
    COMPLETED_STAGE: (
        "No immediate action; enrol for routine reminders for next due visit"),
}
CADRE = {"Medicine not collected": "ASHA", "Test not completed": "ASHA",
         "Review not attended": "CHO", COMPLETED_STAGE: "CHO"}


def priority_score(risk, stage):
    urgency = STAGE_URGENCY.get(stage, 0.5)
    consequence = urgency  # clinical consequence proxy: untreated episode severity
    feasibility = {"Medicine not collected": 1.0, "Test not completed": 0.7,
                   "Review not attended": 0.8, COMPLETED_STAGE: 0.5}[stage]
    return (WEIGHTS["risk"] * risk + WEIGHTS["urgency"] * urgency
            + WEIGHTS["consequence"] * consequence + WEIGHTS["feasibility"] * feasibility)


def tier_from_validation(valid_scores, capacity):
    """Return score thresholds so that ~capacity shares of *all* episodes land
    in each tier when outreach targets predicted dropouts."""
    s = np.sort(valid_scores)[::-1]
    n = len(s)
    p1_end = s[int(capacity["P1"] * n) - 1] if n else 0
    p2_end = s[int((capacity["P1"] + capacity["P2"]) * n) - 1] if n else 0
    p3_end = s[int((capacity["P1"] + capacity["P2"] + capacity["P3"]) * n) - 1] if n else 0
    return {"P1": p1_end, "P2": p2_end, "P3": p3_end}


def system_barrier_flags(ev, F):
    """Recent facility stock-out pressure (known at consult time)."""
    st = ev["stock"]
    last6 = {}
    for fid, g in st.groupby("facility_id"):
        g = g.sort_values("snapshot_month").tail(6)
        last6[fid] = float(pd.to_numeric(g["stockout_flag_month"], errors="coerce").mean())
    return F["facility_id"].map(last6).fillna(0.0)


def main():
    # ---------------- validation-based thresholds -----------------------------
    with open(os.path.join(MODELS, "dropout_model.pkl"), "rb") as f:
        dm = pickle.load(f)
    with open(os.path.join(MODELS, "stage_model.pkl"), "rb") as f:
        sm = pickle.load(f)

    F = pd.read_csv(os.path.join(ART, "features.csv"))
    from train import add_derived, stage_mask
    F = add_derived(F)
    dev = F[(F.cohort == "DEVELOPMENT") & F.lost_to_followup_label.notna()].copy()
    train = dev[dev.consult_date <= pd.Timestamp(DEV_END)]
    valid = dev[dev.consult_date >= pd.Timestamp(VALID_START)]

    # refit chosen model on ALL development data for evaluation scoring
    model_name = dm["model_name"]
    import train as T
    cols = dm["columns"]
    Xdev = encode(dev, T._expand_sets()["D_all"]).reindex(columns=cols, fill_value=0.0)
    full_model = T.pipe(model_name, dm["hyperparams"].get(model_name, {}))
    full_model.fit(Xdev.values, dev["lost_to_followup_label"].astype(int).values)

    Xval = encode(valid, T._expand_sets()["D_all"]).reindex(columns=cols, fill_value=0.0)
    p_val = full_model.predict_proba(Xval.values)[:, 1]

    # predicted stage for ALL validation episodes (stage model gated by risk),
    # used only to calibrate priority-tier thresholds at the configured capacity
    stage_cols = sm["columns"]
    classes = sm["classes"]
    thr_risk = float(np.median(p_val))
    class_names = np.array([ID2STAGE[int(c)] for c in classes])
    sp_all = sm["pipeline"].predict_proba(
        encode(valid, T._expand_sets()["D_all"]).reindex(columns=stage_cols, fill_value=0.0).values)
    stg_all_val = np.where(p_val >= thr_risk,
                           class_names[np.where(stage_mask(valid, classes),
                                                sp_all, -np.inf).argmax(1)],
                           COMPLETED_STAGE)
    scores_val = np.array([priority_score(r, s) for r, s in zip(p_val, stg_all_val)])
    th = tier_from_validation(scores_val, CAPACITY)
    print("[priority] validation thresholds:", {k: round(v, 4) for k, v in th.items()},
          "capacity:", CAPACITY)

    # ---------------- score EVALUATION episodes -------------------------------
    data = load_all()
    aud = pd.read_csv(os.path.join(ART, "linkage_audit.csv"))
    ep = build_episodes(data)
    ep = attach_linkage(ep, data, aud)
    ev = canonicalize(data, aud)
    ev["stock"] = prepare_stock(data)
    ref, fac_lite = build_reference_info(data)
    Fe = build_features(ep, data, ev, ref, fac_lite)
    Fe = add_derived(Fe)
    evalF = Fe[Fe.cohort == "EVALUATION"].copy()
    Xev = encode(evalF, T._expand_sets()["D_all"]).reindex(columns=cols, fill_value=0.0)
    p_ev = full_model.predict_proba(Xev.values)[:, 1]

    Xev_s = encode(evalF, T._expand_sets()["D_all"]).reindex(columns=stage_cols, fill_value=0.0)
    sp_ev = sm["pipeline"].predict_proba(Xev_s.values)
    mask_ev = stage_mask(evalF, classes)
    sp_ev_m = np.where(mask_ev, sp_ev, -np.inf)
    stg_ev = class_names[sp_ev_m.argmax(1)]
    stg_final = np.where(p_ev >= thr_risk, stg_ev, COMPLETED_STAGE)

    # ---------------- reasons (interpretable, non-causal wording) --------------
    stock_pressure = system_barrier_flags(ev, evalF)
    reasons = []
    for _, r in evalF.iterrows():
        bits = []
        d = pd.to_numeric(r["distance_km"], errors="coerce")
        if pd.notna(d) and d >= evalF["distance_km"].quantile(0.75):
            bits.append(f"long facility distance ({d:.0f} km)")
        if r["medicine_advised"] == 1 and r["f_low_stock_share"] >= 0.5:
            bits.append("facility medicine availability pressure")
        if pd.notna(r["hist_disp_rate"]) and r["hist_disp_rate"] < 0.5:
            bits.append("previous low medicine collection")
        if pd.notna(r["hist_review_rate"]) and r["hist_review_rate"] < 0.5:
            bits.append("previous missed reviews")
        if r["prior_outreach_unreachable"] and r["prior_outreach_unreachable"] > 0:
            bits.append("previously unreachable on outreach")
        if r["test_advised"] == 1 and r["hist_test_rate"] is not None and \
                pd.notna(r["hist_test_rate"]) and r["hist_test_rate"] < 0.5:
            bits.append("previous incomplete tests")
        if not bits:
            bits.append("advised care components with limited follow-up history")
        reasons.append("; ".join(bits[:3]))
    evalF["reason"] = reasons
    evalF["system_barrier"] = ((stg_final == "Medicine not collected") &
                               (stock_pressure.values >= 0.33)).astype(int)

    scores_ev = np.array([priority_score(r, s) for r, s in zip(p_ev, stg_final)])
    tiers = np.where(scores_ev >= th["P1"], "P1",
             np.where(scores_ev >= th["P2"], "P2",
              np.where(scores_ev >= th["P3"], "P3", "P4")))

    # ---------------- submission files ---------------------------------------
    sub_dir = SUB
    ep_pred = pd.DataFrame({
        "episode_id": evalF["episode_id"],
        "risk_probability": np.round(p_ev, 4),
        "predicted_lost_to_followup": (p_ev >= 0.5).astype(int),
        # dataset-native stage strings for judge readability
        "predicted_dropout_stage": stg_final,
        "priority_tier": tiers,
    })
    ep_pred.to_csv(os.path.join(sub_dir, "submission_episode_predictions.csv"), index=False)

    # action queue only for episodes routed to outreach (P1-P3)
    q = ep_pred[ep_pred.priority_tier != "P4"].merge(
        evalF[["episode_id", "patient_id", "reason", "system_barrier"]],
        on="episode_id", how="left")
    act_rows = []
    for _, r in q.iterrows():
        # predicted_dropout_stage already holds the long dataset stage name
        stage_long = r["predicted_dropout_stage"]
        action = ACTIONS[stage_long]
        if r.get("system_barrier", 0) == 1:
            action = ("FACILITY ESCALATION FIRST: verify medicine stock at facility; "
                      "then " + action)
            stage_long = "Medicine not collected (system barrier suspected)"
        act_rows.append({
            "episode_id": r["episode_id"],
            "predicted_patient_id": r["patient_id"],
            "priority": r["priority_tier"],
            "reason": r["reason"],
            "recommended_action": action,
            "assigned_cadre": CADRE.get(stage_long.split(" (")[0], "ASHA"),
        })
    pd.DataFrame(act_rows).to_csv(os.path.join(sub_dir, "submission_action_queue.csv"),
                                  index=False)

    # linkage submission: one row per (system, source_patient_id).
    # Policy: if several audit rows map the same source id to different
    # canonical patients (0.04% of ids), the majority mapping wins and the
    # confidence is downgraded to 'low' (ambiguous identity). The per-record
    # trail remains in artifacts/linkage_audit.csv.
    lk_full = pd.read_csv(os.path.join(ART, "submission_linkage.csv"))
    lk_conflict = lk_full.groupby(["source_system", "source_patient_id"])[
        "predicted_patient_id"].transform("nunique") > 1
    lk_mode = lk_full.groupby(["source_system", "source_patient_id"])[
        "predicted_patient_id"].agg(lambda s: s.mode().iat[0])
    conf_rank = {"high": 2, "medium": 1, "low": 0}
    lk_out = (lk_full.assign(_c=lk_full["confidence"].map(conf_rank))
              .sort_values("_c", ascending=False)
              .drop_duplicates(["source_system", "source_patient_id"]))
    lk_out["predicted_patient_id"] = [lk_mode.get((s, p), p) for s, p in
                                       zip(lk_out["source_system"], lk_out["source_patient_id"])]
    lk_out.loc[lk_conflict[lk_out.index], "confidence"] = "low"
    lk_out = lk_out[["source_system", "source_patient_id", "predicted_patient_id",
                     "confidence"]]
    lk_out.to_csv(os.path.join(sub_dir, "submission_linkage.csv"), index=False)

    # ---------------- validation report ---------------------------------------
    linked = lk_out
    rep = {
        "episode_rows": int(len(ep_pred)),
        "episode_ids_unique": bool(ep_pred.episode_id.is_unique),
        "risk_in_0_1": bool(ep_pred.risk_probability.between(0, 1).all()),
        "stages_valid": bool(ep_pred.predicted_dropout_stage.isin(set(STAGES)).all()),
        "tiers_valid": bool(ep_pred.priority_tier.isin(["P1", "P2", "P3", "P4"]).all()),
        "action_rows": int(len(act_rows)),
        "action_episode_ids_unique": bool(pd.Series([a['episode_id'] for a in act_rows]).is_unique),
        "action_all_valid_patients": bool(q["patient_id"].notna().all()),
        "linkage_rows": int(len(linked)),
        "linkage_unique_source_ids": int(linked[["source_system", "source_patient_id"]]
                                         .drop_duplicates().shape[0]),
        "linkage_conf_valid": bool(linked.confidence.isin(["high", "medium", "low"]).all()),
        "tier_shares": ep_pred.priority_tier.value_counts(normalize=True).round(3).to_dict(),
        "stage_shares": ep_pred.predicted_dropout_stage.value_counts(normalize=True).round(3).to_dict(),
    }
    json.dump(rep, open(os.path.join(ART, "submission_validation.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
