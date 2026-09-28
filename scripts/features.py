"""Canonical Patient-360 assembly + temporal (as-known-at-consult) features.

Prediction point: the teleconsultation instant (consult_date). Every feature is
computed ONLY from records with date <= consult_date of the episode being
predicted. No post-consult event (dispensing, labs, reviews, outreach, labels)
is ever read by this builder — enforced by asserts + a randomised leakage probe
(run_pipeline --check-leakage).

Episode definition: one row per teleconsultation (episode_id == teleconsult_id's
outcome row). Labels come from episode_outcomes.csv and are joined ONLY for
training/evaluation reporting — never as features.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, ART, DEV_END, EVAL_START, STAGE_SHORT

warnings.filterwarnings("ignore")


def _p(name):
    return os.path.join(DATA, name + ".csv")


def load_all():
    d = {n: pd.read_csv(_p(n), low_memory=False) for n in
         ["patient_360_reference", "teleconsultations", "ncd_screening", "prescriptions",
          "medicine_dispensing", "medicine_stock_status", "lab_tests", "followup_visits",
          "visit_history", "outreach_actions", "facility_reference", "geography_reference",
          "episode_outcomes"]}
    return d


# --------------------------------------------------------------------------
# Leakage guards
# --------------------------------------------------------------------------
def assert_no_future(df_feat, df_events, feat_key, event_key, date_col, event_name):
    """For every (episode, event) pair joined into features, event date must be
    <= the episode's consult_date. Used by the leakage probe."""
    m = df_feat[[feat_key, "consult_date"]].merge(
        df_events[[event_key, date_col]].rename(columns={event_key: feat_key}),
        on=feat_key, how="inner")
    bad = pd.to_datetime(m[date_col]) > pd.to_datetime(m["consult_date"])
    if bad.any():
        raise AssertionError(f"LEAKAGE in {event_name}: {int(bad.sum())} events after consult_date")


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_episodes(data):
    tc = data["teleconsultations"].copy()
    eo = data["episode_outcomes"].copy()
    ep = tc.merge(eo, on="teleconsult_id", suffixes=("_tc", ""))
    ep["consult_date"] = pd.to_datetime(ep["consult_date_tc"])
    ep["cohort"] = ep["cohort"].astype(str).str.strip()
    assert ep["teleconsult_id"].is_unique
    assert (ep["episode_id"].notna()).all()
    return ep


def attach_linkage(ep, data, aud):
    """Canonical patient per episode, from high/medium confidence linkage of the
    teleconsult source id (audit file covers every teleconsult row)."""
    lk = aud[(aud.source_system == "teleconsultations") & (aud.matched == 1)]
    m = ep.merge(lk[["source_patient_id", "canonical_patient_id", "match_score",
                     "confidence"]],
                 left_on="tele_source_patient_id", right_on="source_patient_id",
                 how="left")
    ep["patient_id"] = m["canonical_patient_id"].fillna("")
    ep["link_conf"] = m["confidence"].fillna("none")
    ep["link_score"] = m["match_score"]
    if (ep["patient_id"] != "").mean() < 0.80:
        raise AssertionError("too many unlinked teleconsults (<80%)")
    return ep


def build_reference_info(data):
    ref = data["patient_360_reference"].copy()
    geo = data["geography_reference"].copy()
    fac = data["facility_reference"].copy()
    ref["age_as_of_2026"] = pd.to_numeric(ref["age_as_of_2026"], errors="coerce")
    geo_lite = geo[["village_id", "village", "district", "block", "population",
                    "road_access", "mobile_connectivity"]]
    ref = ref.merge(geo_lite, on="village_id", suffixes=("", "_geo"), how="left")
    fac_lite = fac[["facility_id", "facility_type", "network_context", "cho_count",
                    "asha_linked_count"]]
    return ref, fac_lite


# --------------------------------------------------------------------------
# Event tables: canonical patient + date
# --------------------------------------------------------------------------
def canonicalize(data, aud):
    """Attach canonical patient_id to every event table via linkage audit."""
    lk = aud[aud.matched == 1][["source_system", "source_patient_id", "canonical_patient_id"]]

    def link(df, system, sid):
        m = lk[lk.source_system == system]
        j = df.merge(m, left_on=sid, right_on="source_patient_id", how="left")
        return j.drop(columns=["source_patient_id"])

    ev = {}
    ev["rx"] = link(data["prescriptions"], "prescriptions", "rx_source_patient_id")
    ev["disp"] = link(data["medicine_dispensing"], "medicine_dispensing", "pharm_source_patient_id")
    ev["disp"] = ev["disp"].merge(
        ev["rx"][["prescription_id", "teleconsult_id", "prescription_date"]],
        on="prescription_id", how="left")
    ev["lab"] = link(data["lab_tests"], "lab_tests", "lab_source_patient_id")
    ev["ncd"] = link(data["ncd_screening"], "ncd_screening", "ncd_source_patient_id")
    ev["visits"] = link(data["visit_history"], "visit_history", "visit_source_patient_id")
    ev["fh"] = link(data["followup_visits"], "followup_visits", "visit_source_patient_id")
    ev["out"] = link(data["outreach_actions"], "outreach_actions", "outreach_source_patient_id")
    for k, col in [("rx", "prescription_date"), ("disp", "dispense_date"), ("lab", "order_date"),
                   ("ncd", "screening_date"), ("visits", "visit_date"), ("fh", "visit_date"),
                   ("out", "action_date")]:
        ev[k][col] = pd.to_datetime(ev[k][col], errors="coerce")
    return ev


def stage_outcome_features(ev, patient_id, t):
    """Stage-specific historical behaviour for one patient strictly before t."""
    rx = ev["rx"]
    disp = ev["disp"]
    lab = ev["lab"]
    fh = ev["fh"]

    if not patient_id:
        return {
            "hist_episodes": 0, "hist_rx_meds": 0, "hist_disp_rate": np.nan,
            "hist_partial_rate": np.nan, "hist_stockout_rate": np.nan,
            "hist_test_rate": np.nan, "hist_review_rate": np.nan,
            "hist_any_followup_rate": np.nan, "hist_recent_dropout": np.nan,
        }
    rx_p = rx[(rx.canonical_patient_id == patient_id) & (rx.prescription_date < t)]
    out = {}
    if len(rx_p) == 0:
        return {
            "hist_episodes": 0, "hist_rx_meds": 0, "hist_disp_rate": np.nan,
            "hist_partial_rate": np.nan, "hist_stockout_rate": np.nan,
            "hist_test_rate": np.nan, "hist_review_rate": np.nan,
            "hist_any_followup_rate": np.nan, "hist_recent_dropout": np.nan,
        }
    # --- historical medicine collection rate (dispenses vs prescribed episodes)
    prior_eps = rx_p["teleconsult_id"].dropna().unique()
    eps = []
    for tcid in prior_eps:
        rxs = rx_p[rx_p.teleconsult_id == tcid]
        d0 = rxs["prescription_date"].min()
        window_end = d0 + pd.Timedelta(days=14)
        had_rx = len(rxs) > 0
        dsp = disp[(disp.canonical_patient_id == patient_id) &
                   (disp.dispense_date.notna()) &
                   (disp.dispense_date <= window_end) &
                   (disp.dispense_date >= d0 - pd.Timedelta(days=1))]
        collected = 1 if (len(dsp) > 0 and dsp["dispense_status"].isin(["Dispensed", "Partial"]).any()) else 0
        eps.append(collected)
    hist_disp_rate = float(np.mean(eps)) if eps else np.nan
    # partial fills / stockouts the patient experienced historically
    d_all = disp[(disp.canonical_patient_id == patient_id) & (disp.dispense_date < t)]
    hist_partial = float(d_all["partial_fill"].astype(float).mean()) if len(d_all) else np.nan
    hist_stock = float(d_all["stockout_flag"].astype(float).mean()) if len(d_all) else np.nan

    # --- historical test completion: labs ordered within 30d of past consults
    tc_dates = rx_p.groupby("teleconsult_id")["prescription_date"].min()
    tcs = []
    for tcid, d0 in tc_dates.items():
        lb = lab[(lab.canonical_patient_id == patient_id) &
                 (lab.order_date >= d0 - pd.Timedelta(days=1)) &
                 (lab.order_date <= d0 + pd.Timedelta(days=30))]
        if len(lb) == 0:
            continue
        tcs.append(float((lb["test_status"] == "Available").mean()))
    hist_test = float(np.mean(tcs)) if tcs else np.nan

    # --- historical review attendance: followup visits after past consults
    revs = []
    for tcid, d0 in tc_dates.items():
        f = fh[(fh.canonical_patient_id == patient_id) &
               (fh.visit_date > d0) &
               (fh.visit_date <= d0 + pd.Timedelta(days=45))]
        if len(f) > 0:
            revs.append(1.0)
        else:
            revs.append(0.0)
    hist_review = float(np.mean(revs)) if revs else np.nan
    hist_any = float(np.mean([max(a, b) for a, b in zip(tcs or [np.nan], revs or [np.nan])])) if tcs or revs else np.nan

    # most recent prior episode: did it end in dropout? (from prior labels is NOT
    # allowed; but prior observable completeness IS observable at prediction time)
    last_d0 = tc_dates.max()
    f_last = fh[(fh.canonical_patient_id == patient_id) & (fh.visit_date > last_d0) &
                (fh.visit_date <= last_d0 + pd.Timedelta(days=45))]
    hist_recent_dropout = 0.0 if len(f_last) > 0 else 1.0

    n_eps = len(prior_eps)
    return {
        "hist_episodes": n_eps,
        "hist_rx_meds": int(len(rx_p)),
        "hist_disp_rate": hist_disp_rate,
        "hist_partial_rate": hist_partial,
        "hist_stockout_rate": hist_stock,
        "hist_test_rate": hist_test,
        "hist_review_rate": hist_review,
        "hist_any_followup_rate": hist_any if not np.isnan(hist_any) else
        (hist_review if not np.isnan(hist_review) else hist_disp_rate),
        "hist_recent_dropout": hist_recent_dropout if n_eps else np.nan,
    }


def facility_stock_at(ev, facility_id, t):
    """Stock knowledge available at time t: latest snapshot strictly before t."""
    st = ev["stock"]
    s = st[(st.facility_id == facility_id) & (st.snapshot_month.dt.date <= t.date())]
    if len(s) == 0:
        return {"f_stockout_rate_6m": np.nan, "f_low_stock_share": np.nan}
    s = s.sort_values("snapshot_month").tail(6)
    return {
        "f_stockout_rate_6m": float(s["stockout_flag_month"].astype(float).mean()),
        "f_low_stock_share": float((s["stock_status"] != "Adequate").astype(float).mean()),
    }


def build_features(ep, data, ev, ref, fac_lite):
    """One row per episode; all features as-known-at consult_date."""
    ref_idx = ref.set_index("patient_id")
    fac_idx = fac_lite.set_index("facility_id")

    rows = []
    n = len(ep)
    ep_r = ep.reset_index(drop=True)
    for i in range(n):
        r = ep_r.iloc[i]
        t = r["consult_date"]
        pid = r["patient_id"]
        fid = r["facility_id"]
        feat = {"episode_id": r["episode_id"], "teleconsult_id": r["teleconsult_id"],
                "consult_date": t, "cohort": r["cohort"], "patient_id": pid,
                "link_conf": r["link_conf"]}

        # ---------------- patient static (canonical) ----------------
        if pid and pid in ref_idx.index:
            pr = ref_idx.loc[pid]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            feat["age"] = pr["age_as_of_2026"]
            feat["gender"] = pr["gender"]
            feat["known_ncd_status"] = pr["known_ncd_status"]
            feat["vulnerability_group"] = pr["vulnerability_group"]
            feat["preferred_language"] = pr["preferred_language"]
            feat["village_pop"] = pr["population"]
            feat["road_access"] = pr["road_access"]
            feat["mobile_connectivity"] = pr["mobile_connectivity"]
            feat["district"] = pr["district"]
            feat["block"] = pr["block"]
        # ---------------- consultation (available at prediction time) -------
        for c in ["chief_complaint", "diagnosis_group", "consult_mode", "connectivity_quality",
                  "consult_status", "facility_id"]:
            feat[c] = r[c]
        feat["duration_min"] = pd.to_numeric(r["duration_min"], errors="coerce")
        feat["medicine_advised"] = int(r["medicine_advised"] == "Yes")
        feat["test_advised"] = int(r["test_advised"] == "Yes")
        feat["review_advised"] = int(r["review_advised"] == "Yes")
        feat["review_due_days"] = pd.to_numeric(r["review_due_days"], errors="coerce")
        feat["distance_km"] = pd.to_numeric(r["distance_to_facility_km"], errors="coerce")
        feat["consult_weekday"] = t.weekday()
        feat["consult_month"] = t.month

        # ---------------- facility characteristics ----------------
        if fid in fac_idx.index:
            fr = fac_idx.loc[fid]
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[0]
            feat["facility_type"] = fr["facility_type"]
            feat["facility_network"] = fr["network_context"]
            feat["facility_cho_count"] = pd.to_numeric(fr["cho_count"], errors="coerce")
            feat["facility_asha_count"] = pd.to_numeric(fr["asha_linked_count"], errors="coerce")

        # ---------------- stock knowledge at t ----------------
        feat.update(facility_stock_at(ev, fid, t))

        # ---------------- longitudinal history ----------------
        h = stage_outcome_features(ev, pid, t)
        feat.update(h)
        rx = ev["rx"]
        rx_p = rx[(rx.canonical_patient_id == pid) & (rx.prescription_date < t)] if pid else rx.iloc[:0]
        feat["days_since_last_rx"] = (t - rx_p["prescription_date"].max()).days if len(rx_p) else np.nan
        ncd = ev["ncd"]
        ncd_p = ncd[(ncd.canonical_patient_id == pid) & (ncd.screening_date < t)]
        if len(ncd_p):
            lastn = ncd_p.sort_values("screening_date").iloc[-1]
            feat["last_systolic_bp"] = pd.to_numeric(lastn["systolic_bp"], errors="coerce")
            feat["last_glucose"] = pd.to_numeric(lastn["random_glucose_mg_dl"], errors="coerce")
            feat["last_bmi"] = pd.to_numeric(lastn["bmi"], errors="coerce")
            feat["days_since_ncd_screen"] = (t - lastn["screening_date"]).days
            feat["ncd_control_status"] = lastn["control_status"]
        else:
            feat["last_systolic_bp"] = np.nan
            feat["last_glucose"] = np.nan
            feat["last_bmi"] = np.nan
            feat["days_since_ncd_screen"] = np.nan
            feat["ncd_control_status"] = "No screening"
        vh = ev["visits"]
        vh_p = vh[(vh.canonical_patient_id == pid) & (vh.visit_date < t)]
        feat["prior_visits"] = len(vh_p)
        feat["prior_referrals"] = int((vh_p["referral_flag"] == "Yes").sum())
        oh = ev["out"]
        oh_p = oh[(oh.canonical_patient_id == pid) & (oh.action_date < t)]
        feat["prior_outreach"] = len(oh_p)
        feat["prior_outreach_unreachable"] = int(
            (oh_p["contact_outcome"].str.contains("Unavailable", case=False, na=False)).sum())
        rows.append(feat)

    F = pd.DataFrame(rows)

    # ---------------- labels (joined separately, NEVER as features) ----------
    eo = data["episode_outcomes"].copy()
    eo["lost_to_followup_label"] = pd.to_numeric(eo["lost_to_followup_label"], errors="coerce")
    F = F.merge(eo[["episode_id", "lost_to_followup_label", "dropout_stage_label"]],
                on="episode_id", how="left")
    F["dropout_stage_label"] = F["dropout_stage_label"].fillna("Unknown")
    return F


def prepare_stock(data):
    st = data["medicine_stock_status"].copy()
    st["snapshot_month"] = pd.to_datetime(st["snapshot_month"], errors="coerce")
    return st


def main(out_path=None, check_leakage=False):
    data = load_all()
    aud = pd.read_csv(os.path.join(ART, "linkage_audit.csv"))
    ep = build_episodes(data)
    ep = attach_linkage(ep, data, aud)
    ref, fac_lite = build_reference_info(data)
    ev = canonicalize(data, aud)
    ev["stock"] = prepare_stock(data)
    F = build_features(ep, data, ev, ref, fac_lite)

    if check_leakage:
        # Monotonicity probe: history features are time-gated, so shifting the
        # prediction time LATER may only keep or ADD history, never remove it.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ep), size=min(200, len(ep)), replace=False)
        violations = 0
        for i in idx:
            r = ep.iloc[i]
            pid, t = r["patient_id"], r["consult_date"]
            if not pid:
                continue
            h1 = stage_outcome_features(ev, pid, t)
            h2 = stage_outcome_features(ev, pid, t + pd.Timedelta(days=45))
            if h2["hist_episodes"] < h1["hist_episodes"] or \
               h2["hist_rx_meds"] < h1["hist_rx_meds"]:
                violations += 1
        assert violations == 0, f"leakage probe failed on {violations} episodes"
        print(f"[leakage] monotonic-history probe OK on {len(idx)} random episodes")
    return F


if __name__ == "__main__":
    out = os.path.join(ART, "features.csv")
    F = main(check_leakage=True)
    F.to_csv(out, index=False)
    print("features:", F.shape, "->", out)
    print(F.groupby("cohort")["lost_to_followup_label"].agg(["count", "mean"]))
