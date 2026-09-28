"""Shared paths and configuration for the care-journey assurance pipeline."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "hackathon", "Infinum_2026_Candidate_Dataset_Pack")
ART = os.path.join(ROOT, "artifacts")
FIG = os.path.join(ART, "figures")
MODELS = os.path.join(ART, "models")
SUB = os.path.join(ROOT, "submission")
for d in (ART, FIG, MODELS, SUB):
    os.makedirs(d, exist_ok=True)

# ---------------- temporal configuration ----------------
DEV_END = "2026-06-30"        # DEVELOPMENT cohort: consult_date <= 30-Jun-2026
EVAL_START = "2026-07-01"     # EVALUATION cohort: consult_date 01-Jul..15-Aug
TRAIN_END = "2026-04-30"      # temporal train: consultations up to 30-Apr
VALID_START = "2026-05-01"    # temporal validation: 01-May..30-Jun

# ---------------- care journey stages ----------------
STAGES = ["Medicine not collected", "Test not completed", "Review not attended",
          "Completed care journey"]
STAGE2ID = {s: i for i, s in enumerate(STAGES)}
ID2STAGE = {i: s for i, s in enumerate(STAGES)}
COMPLETED_STAGE = "Completed care journey"

STAGE_SHORT = {
    "Medicine not collected": "medicine",
    "Test not completed": "test",
    "Review not attended": "review",
    "Completed care journey": "completed",
}

# ---------------- linkage decision thresholds (documented, auditable) ------
TH_AUTO_HIGH = 0.90      # auto-accept with margin >= 0.05
MARGIN_HIGH = 0.05
TH_AUTO_MED = 0.78       # auto-accept with margin >= 0.12
MARGIN_MED = 0.12
TH_REVIEW = 0.55         # below this -> unmatched
MIN_NAME_SIM_FOR_PHONE_ONLY = 0.45  # phone-4 matches must still corroborate on name

SEED = 42
