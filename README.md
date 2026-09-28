# Care-Journey Assurance — Stage-Aware Dropout Prediction (INFINUM 2026)

**From dropout prediction to care-journey assurance.** Patients do not simply
"drop out" — they fail at specific points of a care journey (medicine collection →
tests → review). This system predicts, immediately after a teleconsultation,
**whether** an episode will be lost to follow-up **and where** it will most
likely fail, then converts that into a prioritised, stage-specific ASHA/CHO
action queue.

---

## How to run it

You need Python 3.11 on Windows (any OS works; commands below are Windows-style
since that is what this was built on).

```bash
# 1. one-time setup — create a virtual env and install dependencies
python -m venv .venv
.venv\Scripts\python -m pip install pandas numpy scikit-learn xgboost lightgbm shap pypdf matplotlib rapidfuzz

# 2. run the whole solution (about 4 minutes on a laptop)
.venv\Scripts\python run_pipeline.py
```

That single command executes everything in order and prints progress:

| step | what it does | output |
|---|---|---|
| linkage | fuzzy-matches 42,900 register records to 5,000 canonical patients | `submission/submission_linkage.csv`, `artifacts/linkage_audit.csv` |
| features | builds one row per teleconsultation using **only** information known at the consult instant; runs a leakage probe | `artifacts/features.csv` |
| training | compares 5 models with a temporal split, tunes on an inner split, compares stage approaches A vs B, runs ablations, SHAP, subgroup checks | `artifacts/model_comparison.csv`, `ablation.csv`, `stage_comparison.csv`, figures |
| submissions | scores the evaluation cohort, calibrates priority tiers to outreach capacity, writes the three CSVs, validates structure | `submission/*.csv`, `artifacts/submission_validation.json` |
| docs & figures | EDA charts, data dictionary, example patient explanations | `docs/`, `artifacts/figures/` |

Prefer to run pieces manually? Each script is standalone:

```bash
.venv\Scripts\python scripts/linkage.py
.venv\Scripts\python scripts/features.py
.venv\Scripts\python scripts/train.py
.venv\Scripts\python scripts/action_queue.py
.venv\Scripts\python scripts/figures.py
.venv\Scripts\python scripts/make_data_dictionary.py
.venv\Scripts\python scripts/explain_examples.py
```

Everything is deterministic (fixed seeds, temporal splits, no random shuffling),
so re-running produces identical numbers.

## The judge-test answers (short form)

| Judge question | Where / answer |
|---|---|
| Problem defined? | `docs/RESULTS_REPORT.md` §1 — label = `lost_to_followup_label` (1 iff any advised component incomplete), stage = first unresolved of medicine/test/review |
| More than a classifier? | Two-headed engine: risk model + gated stage model + priority engine + action mapping (`scripts/action_queue.py`) |
| Novelty? | Stage-aware prediction with structural masking (advised-pattern constraints) + system-barrier flagging |
| Defensible methodology? | Temporal train/valid split (≤2026-04-30 / 05-01→06-30), hyperparameters tuned on an *inner* temporal split, evaluation cohort untouched |
| Temporal leakage prevented? | Every feature time-gated at consult instant; monotonicity probe (`features.py --check-leakage` path); `docs/data_dictionary.md` marks every column |
| Interpretable? | SHAP summary + per-patient explanation cards (`docs/example_explanations.md`) with associational wording |
| Prediction → action? | Stage-specific intervention text + cadre assignment in `submission/submission_action_queue.csv` |
| System barriers? | Recent facility stock-out pressure escalates *facility* follow-up instead of labelling the patient non-adherent |
| ASHA/CHO feasible? | One line per patient: reason in plain language, one action, one cadre |
| Scales? | Pure tabular pipeline; <2 min end-to-end on a laptop; `run_pipeline.py` |
| Impact measurable? | Queue design enables A/B measurement (outreach vs control) — no unmeasured outcome claims made |
| Limitations honest? | `docs/RESULTS_REPORT.md` §8 |

## Headline results (temporal validation, never seen in tuning)

| model | ROC-AUC | PR-AUC | F1 | P@10% | R@20% |
|---|---|---|---|---|---|
| **Random Forest (chosen)** | **0.705** | **0.665** | **0.634** | **0.808** | **0.301** |
| Logistic regression | 0.692 | 0.653 | 0.609 | 0.758 | 0.294 |
| LightGBM / XGBoost / GradBoost | 0.67 | 0.61–0.62 | 0.60–0.62 | 0.70 | 0.27–0.28 |
| baseline (prevalence) | 0.500 | 0.473 | — | 0.453 | 0.194 |

- **Boosting lost to a tuned random forest** — we kept the simpler winner (no deep learning).
- **Stage prediction**: hierarchical (Approach B) macro-F1 **0.560** vs direct multiclass (A) **0.247** → B selected. Confusion matrix: `artifacts/stage_confusion_matrix.csv`.
- **Ablations**: demographics alone ≈ chance (PR 0.476) → +consultation 0.661 → +history 0.664 → +operational 0.665. Consultation content is the big lever; history/stock add smaller, consistent gains.
- **Linkage**: 89.6% of 42,900 source records auto-linked, 90% high-confidence, 4,906/5,000 patients, 93.2% mobile-transitivity consistency.

## Repo layout

```
run_pipeline.py              one-command reproduction
scripts/
  linkage.py                 entity linkage + submission_linkage.csv + audit
  features.py                temporal (as-known-at-consult) feature builder + leakage probe
  train.py                   model comparison, stage A/B, ablations, SHAP, fairness
  action_queue.py            priority engine + 3 submission CSVs + validation
  figures.py                 EDA + linkage figures
  make_data_dictionary.py    docs/data_dictionary.md with per-column verdicts
  explain_examples.py        per-patient explanation cards
docs/                        data dictionary, results report, presentation, cards
artifacts/                   all metrics, figures, models (reproducible outputs)
submission/                  FINAL: submission_linkage.csv, submission_episode_predictions.csv, submission_action_queue.csv
hackathon/                   provided pack (untouched)
```

## Ethics & governance

Synthetic data, but the design follows production discipline: minimum-necessary
features (no free-text identity data leaves the linkage layer), role-based
delivery (cadre assignment), explainable reasons for every queued action,
explicit escalation path for system barriers, and an audit file for every
linkage decision (`artifacts/linkage_audit.csv`).
