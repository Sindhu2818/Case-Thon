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

ACTIONS_HINDI = {
    "Medicine not collected": (
        "48 घंटे के भीतर मरीज से संपर्क करें; अस्पताल में दवा की उपलब्धता की पुष्टि करें; "
        "दवा लेने में सहायता करें; यदि दवा नहीं है, तो सुविधा प्रभारी को सूचित करें (सिस्टम बाधा)।"
    ),
    "Test not completed": (
        "टेस्ट के विवरण और स्थान की जानकारी के लिए कॉल करें; शेड्यूलिंग में सहायता करें; "
        "7 दिनों के बाद पुनः जांच करें।"
    ),
    "Review not attended": (
        "समीक्षा की तारीख से पहले याद दिलाएं; स्लॉट बुक करने में मदद करें; "
        "समीक्षा मिस होने के 3 दिनों के भीतर फॉलो-अप करें।"
    ),
    COMPLETED_STAGE: "कोई तत्काल कार्रवाई नहीं; अगले विजिट के लिए रिमाइंडर्स में नामांकित करें।"
}

CADRE = {"Medicine not collected": "ASHA", "Test not completed": "ASHA",
         "Review not attended": "CHO", COMPLETED_STAGE: "CHO"}

# Capacity constraints per facility (Load balancing upgrade)
MAX_TASKS_PER_FACILITY = 10


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
    # Use evaluation set from existing features.csv rather than raw data!
    evalF = F[F.cohort == "EVALUATION"].copy()
    
    # ensure it's not empty
    if len(evalF) == 0:
        raise ValueError("Evaluation data missing from features.csv")
    
    Xev = encode(evalF, T._expand_sets()["D_all"]).reindex(columns=cols, fill_value=0.0)
    p_ev = full_model.predict_proba(Xev.values)[:, 1]

    Xev_s = encode(evalF, T._expand_sets()["D_all"]).reindex(columns=stage_cols, fill_value=0.0)
    sp_ev = sm["pipeline"].predict_proba(Xev_s.values)
    mask_ev = stage_mask(evalF, classes)
    sp_ev_m = np.where(mask_ev, sp_ev, -np.inf)
    stg_ev = class_names[sp_ev_m.argmax(1)]
    stg_final = np.where(p_ev >= thr_risk, stg_ev, COMPLETED_STAGE)

    # ---------------- reasons (interpretable, non-causal wording) --------------
    # Calculate stock pressure proxy directly from feature instead of using raw records
    # f_low_stock_share or f_stockout_rate_6m works!
    stock_pressure = evalF["f_stockout_rate_6m"].fillna(0.0)
    
    reasons = []
    reasons_hi = []
    for _, r in evalF.iterrows():
        bits = []
        bits_hi = []
        d = pd.to_numeric(r["distance_km"], errors="coerce")
        if pd.notna(d) and d >= evalF["distance_km"].quantile(0.75):
            bits.append(f"long facility distance ({d:.0f} km)")
            bits_hi.append(f"अस्पताल की अधिक दूरी ({d:.0f} किमी)")
        if r.get("medicine_advised", 0) == 1 and r.get("f_low_stock_share", 0) >= 0.5:
            bits.append("facility medicine availability pressure")
            bits_hi.append("अस्पताल में दवा की कमी")
        if pd.notna(r.get("hist_disp_rate")) and r.get("hist_disp_rate", 1) < 0.5:
            bits.append("previous low medicine collection")
            bits_hi.append("पहले दवा कम लेने का इतिहास")
        if pd.notna(r.get("hist_review_rate")) and r.get("hist_review_rate", 1) < 0.5:
            bits.append("previous missed reviews")
            bits_hi.append("पहले समीक्षा मिस करने का इतिहास")
        if r.get("prior_outreach_unreachable", 0) and r.get("prior_outreach_unreachable", 0) > 0:
            bits.append("previously unreachable on outreach")
            bits_hi.append("पहले आउटरीच पर संपर्क नहीं हो पाया")
        if r.get("test_advised", 0) == 1 and r.get("hist_test_rate") is not None and \
                pd.notna(r.get("hist_test_rate")) and r["hist_test_rate"] < 0.5:
            bits.append("previous incomplete tests")
            bits_hi.append("पहले अधूरे टेस्ट")
        if not bits:
            bits.append("advised care components with limited follow-up history")
            bits_hi.append("सीमित फॉलो-अप इतिहास")
        reasons.append("; ".join(bits[:3]))
        reasons_hi.append("; ".join(bits_hi[:3]))
        
    evalF["reason"] = reasons
    evalF["reason_hindi"] = reasons_hi
    evalF["system_barrier"] = ((stg_final == "Medicine not collected") &
                               (stock_pressure.values >= 0.33)).astype(int)

    scores_ev = np.array([priority_score(r, s) for r, s in zip(p_ev, stg_final)])
    evalF["_priority_score"] = scores_ev # Store for sorting
    
    tiers = np.where(scores_ev >= th["P1"], "P1",
             np.where(scores_ev >= th["P2"], "P2",
              np.where(scores_ev >= th["P3"], "P3", "P4")))

    # ---------------- API/Feedback ready submission files -------------------------
    sub_dir = SUB
    ep_pred = pd.DataFrame({
        "episode_id": evalF["episode_id"],
        "risk_probability": np.round(p_ev, 4),
        "predicted_lost_to_followup": (p_ev >= 0.5).astype(int),
        "predicted_dropout_stage": stg_final,
        "priority_tier": tiers,
    })
    ep_pred.to_csv(os.path.join(sub_dir, "submission_episode_predictions.csv"), index=False)

    # Calculate capacity per facility and overflow
    evalF["priority_tier"] = tiers
    evalF["predicted_dropout_stage"] = stg_final
    q = evalF[evalF["priority_tier"] != "P4"].copy()
    
    # Sort within facility so the highest scores get assigned first
    q = q.sort_values(by=["facility_id", "_priority_score"], ascending=[True, False])
    
    # Add a rank to identify overflow
    q["_rank_in_facility"] = q.groupby("facility_id").cumcount() + 1
    
    act_rows = []
    for _, r in q.iterrows():
        stage_long = r["predicted_dropout_stage"]
        action = ACTIONS.get(stage_long, ACTIONS[COMPLETED_STAGE])
        action_hi = ACTIONS_HINDI.get(stage_long, ACTIONS_HINDI[COMPLETED_STAGE])
        
        if r.get("system_barrier", 0) == 1:
            action = ("FACILITY ESCALATION FIRST: verify medicine stock at facility; "
                      "then " + action)
            action_hi = ("पहले अस्पताल प्रभारी को सूचित करें: सुविधा में दवा के स्टॉक की पुष्टि करें; "
                         "फिर " + action_hi)
            stage_long = "Medicine not collected (system barrier suspected)"

        # Load Balancing Logic
        assigned_cadre = CADRE.get(stage_long.split(" (")[0], "ASHA")
        priority = r["priority_tier"]
        
        if r["_rank_in_facility"] > MAX_TASKS_PER_FACILITY:
            # Re-route overflow to Telecall to prevent ASHA burnout
            assigned_cadre = "TELECALL_OVERFLOW"
            priority = "P4_OVERFLOW"

        act_rows.append({
            "episode_id": r["episode_id"],
            "predicted_patient_id": r["patient_id"],
            "priority": priority,
            "reason": r["reason"],
            "reason_local_lang": r["reason_hindi"],
            "recommended_action": action,
            "recommended_action_local_lang": action_hi,
            "assigned_cadre": assigned_cadre,
            "feedback_status_link": f"https://api.carejourney.example.com/status-update?episode={r['episode_id']}",
        })
        
    pd.DataFrame(act_rows).to_csv(os.path.join(sub_dir, "submission_action_queue.csv"), index=False)

    # We do not have raw linkage_audit anymore for the linkage CSV, so we generate a mock or bypass the linkage audit logic.
    # We can just skip linkage regeneration here since it's an ASHA task upgrade, not linkage upgrade.
    # ---------------- validation report ---------------------------------------
    rep = {
        "episode_rows": int(len(ep_pred)),
        "episode_ids_unique": bool(ep_pred.episode_id.is_unique),
        "risk_in_0_1": bool(ep_pred.risk_probability.between(0, 1).all()),
        "stages_valid": bool(ep_pred.predicted_dropout_stage.isin(set(STAGES)).all()),
        "tiers_valid": bool(ep_pred.priority_tier.isin(["P1", "P2", "P3", "P4"]).all()),
        "action_rows": int(len(act_rows)),
        "action_episode_ids_unique": bool(pd.Series([a['episode_id'] for a in act_rows]).is_unique),
        "action_all_valid_patients": bool(q["patient_id"].notna().all()),
        "tier_shares": ep_pred.priority_tier.value_counts(normalize=True).round(3).to_dict(),
        "stage_shares": ep_pred.predicted_dropout_stage.value_counts(normalize=True).round(3).to_dict(),
    }
    json.dump(rep, open(os.path.join(ART, "submission_validation.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == "__main__":
    main()
