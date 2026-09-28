"""Generate docs/data_dictionary.md: the provided dictionary + profiling stats
+ explicit prediction-use / leakage verdicts per column."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA, ROOT

# Verdict rules: (use_for_prediction, leakage_note)
LABELS = {
    ("episode_outcomes.csv", "lost_to_followup_label"):
        ("LABEL - training only, never a feature",
         "Outcome of the whole episode; defined only after the journey ends."),
    ("episode_outcomes.csv", "dropout_stage_label"):
        ("LABEL (stage) - training only, never a feature",
         "First unresolved stage; post-consult information."),
    ("episode_outcomes.csv", "cohort"):
        ("Split definition only", "Defines DEVELOPMENT vs EVALUATION."),
}

NEVER_USE = {
    ("medicine_dispensing.csv", "dispense_status"): "post-consult outcome (unless historical)",
    ("medicine_dispensing.csv", "dispensed_qty"): "post-consult outcome (unless historical)",
    ("medicine_dispensing.csv", "partial_fill"): "post-consult outcome (unless historical)",
    ("medicine_dispensing.csv", "stockout_flag"): "post-consult event; use monthly ledger instead",
    ("lab_tests.csv", "test_status"): "post-consult outcome (unless historical)",
    ("lab_tests.csv", "result_flag"): "post-consult outcome (unless historical)",
    ("lab_tests.csv", "sample_date"): "post-consult event (unless historical)",
    ("lab_tests.csv", "result_date"): "post-consult event (unless historical)",
    ("followup_visits.csv", "clinical_status"): "post-consult outcome (unless historical)",
    ("visit_history.csv", "clinical_status"): "post-consult outcome (unless historical)",
    ("outreach_actions.csv", "contact_outcome"): "post-consult outcome (unless historical)",
}

ALLOWED_AT_T = {  # recorded during/at the consultation itself
    ("teleconsultations.csv", "medicine_advised"), ("teleconsultations.csv", "test_advised"),
    ("teleconsultations.csv", "review_advised"), ("teleconsultations.csv", "review_due_days"),
    ("teleconsultations.csv", "duration_min"), ("teleconsultations.csv", "connectivity_quality"),
    ("teleconsultations.csv", "consult_status"), ("teleconsultations.csv", "distance_to_facility_km"),
}

HISTORICAL_OK = {
    "dispense_date", "order_date", "visit_date", "action_date", "screening_date",
    "prescription_date", "systolic_bp", "diastolic_bp", "random_glucose_mg_dl", "bmi",
    "ncd_status", "control_status", "opening_stock", "received_qty",
    "dispensed_qty_month", "closing_stock", "stockout_flag_month", "stockout_days",
    "stock_status",
}


def verdicts(table, field, note):
    key = (table, field)
    if key in LABELS:
        return LABELS[key]
    if key in NEVER_USE:
        return ("Only as STRICTLY PRE-consult history", NEVER_USE[key])
    if key in ALLOWED_AT_T:
        return ("Yes - available at prediction time",
                "Recorded during the consultation (part of the encounter).")
    if field in HISTORICAL_OK:
        return ("Only as strictly pre-consult history",
                "Time-gate: date <= consult_date of the predicted episode.")
    if table == "patient_360_reference.csv" or table == "geography_reference.csv" \
            or table == "facility_reference.csv":
        return ("Yes - static reference", "Time-invariant reference data.")
    if table == "teleconsultations.csv":
        return ("Yes - available at prediction time",
                "Recorded at/before the consultation instant.")
    return ("Conditional", "Use only if strictly before the prediction timestamp.")


def main():
    dd = pd.read_csv(os.path.join(DATA, "data_dictionary.csv"))
    dd.columns = [c.strip().lstrip("\ufeff") for c in dd.columns]
    rows = []
    for _, r in dd.iterrows():
        use, leak = verdicts(r["file"], r["field"], r.get("competition_note", ""))
        rows.append({"table": r["file"], "column": r["field"],
                     "meaning": r["description"],
                     "competition_note": r.get("competition_note", "") or "",
                     "use_for_prediction": use, "leakage_risk": leak})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    path = os.path.join(ROOT, "docs", "data_dictionary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Data Dictionary - with prediction-use and leakage verdicts\n\n")
        f.write("Derived from the provided `data_dictionary.csv`; verdicts apply our "
                "prediction-time rule: a feature may enter the model only if it was "
                "recorded at/before the teleconsultation instant of the episode being "
                "scored (or is a time-invariant reference attribute).\n\n")
        f.write("| table | column | meaning | use for prediction | leakage risk |\n")
        f.write("|---|---|---|---|---|\n")
        for _, r in out.iterrows():
            meaning = str(r["meaning"]).replace("|", "/")
            note = str(r["competition_note"]).replace("|", "/")
            if note and note.lower() != "nan":
                meaning += f" *(pack note: {note})*"
            f.write(f"| {r['table']} | `{r['column']}` | {meaning} | "
                    f"{r['use_for_prediction']} | {r['leakage_risk']} |\n")
        f.write("\n## Labels\n\n"
                "- `lost_to_followup_label` = 1 iff any advised component remained "
                "incomplete (matches stage != 'Completed care journey'; dev agreement 100%).\n"
                "- `dropout_stage_label` = first unresolved stage in the order "
                "medicine -> test -> review.\n")
    print("wrote", path, "rows:", len(out))


if __name__ == "__main__":
    main()
