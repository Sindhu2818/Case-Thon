"""Step 1: inspect every dataset, column, and submission template.

Outputs a printed report and saves JSON summary to artifacts/eda/columns.json.
Pure profiling — no modeling assumptions here.
"""
import json
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "hackathon", "Infinum_2026_Candidate_Dataset_Pack")
ART = os.path.join(ROOT, "artifacts", "eda")
os.makedirs(ART, exist_ok=True)

TABLES = [
    "patient_360_reference", "teleconsultations", "ncd_screening", "prescriptions",
    "medicine_dispensing", "medicine_stock_status", "lab_tests", "followup_visits",
    "visit_history", "outreach_actions", "facility_reference", "geography_reference",
    "episode_outcomes",
]
TEMPLATES = [
    "submission_template_linkage",
    "submission_template_episode_predictions",
    "submission_template_action_queue",
]

summary = {}

for t in TABLES + TEMPLATES:
    df = pd.read_csv(os.path.join(DATA, f"{t}.csv"), low_memory=False)
    print("=" * 100)
    print(f"TABLE {t}  shape={df.shape}")
    print("=" * 100)
    info = {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": {}}
    for c in df.columns:
        col = df[c]
        nn = col.dropna()
        entry = {
            "dtype": str(col.dtype),
            "n_missing": int(col.isna().sum()),
            "n_unique": int(nn.nunique()) if len(nn) else 0,
        }
        if len(nn):
            if pd.api.types.is_numeric_dtype(col):
                entry["min"] = float(nn.min())
                entry["max"] = float(nn.max())
                entry["mean"] = float(nn.mean())
                entry["sample"] = [str(x) for x in nn.head(3)]
            else:
                vc = nn.astype(str).value_counts().head(12)
                entry["top_values"] = {str(k): int(v) for k, v in vc.items()}
                entry["sample"] = [str(x) for x in nn.head(3)]
        else:
            entry["sample"] = []
        info["columns"][str(c)] = entry
    print(df.head(3).to_string())
    print()
    summary[t] = info

with open(os.path.join(ART, "columns.json"), "w") as f:
    json.dump(summary, f, indent=1, default=str)

# Key structural checks
tc = pd.read_csv(os.path.join(DATA, "teleconsultations.csv"))
eo = pd.read_csv(os.path.join(DATA, "episode_outcomes.csv"))
print("\n### STRUCTURAL CHECKS")
print("teleconsult rows:", len(tc), " episode_outcome rows:", len(eo))
print("episode_id unique in outcomes:", eo.episode_id.is_unique)
print("teleconsult_id unique in outcomes:", eo.teleconsult_id.is_unique)
print("teleconsult_id unique in teleconsultations:", tc.teleconsult_id.is_unique)
print("cohort counts:", eo.cohort.value_counts().to_dict())
tc_dates = pd.to_datetime(tc.consult_date, errors="coerce")
print("consult date range:", tc_dates.min(), "->", tc_dates.max())
print("DEV consult max:", tc_dates[eo.cohort.eq("DEVELOPMENT")].max())
print("EVAL consult min:", tc_dates[eo.cohort.eq("EVALUATION")].min())
print("label dist (dev):", eo.lost_to_followup_label.value_counts(dropna=False).to_dict())
print("stage dist (dev):", eo.dropout_stage_label.value_counts(dropna=False).to_dict())
print("outcome rows with teleconsult_id not in teleconsultations:",
      (~eo.teleconsult_id.isin(tc.teleconsult_id)).sum())
print("duplicate teleconsult rows by episode_id:",
      tc.merge(eo, on="teleconsult_id").episode_id.duplicated().sum())
