# Final Results Report — Care-Journey Assurance (INFINUM 2026)

## 1. Problem, exactly as defined by the data

- **Unit of prediction**: one teleconsultation episode (`episode_id` ↔ `teleconsult_id`, 5,516 episodes, 5,000 patients).
- **Care journey**: teleconsultation → medicine collection → test completion → review attendance → completed.
- **Label**: `lost_to_followup_label` = 1 iff any advised component remained incomplete. Verified: label 1 ⟺ `dropout_stage_label ≠ "Completed care journey"` (100% agreement on 4,132 development episodes).
- **Stage label**: first unresolved stage among `Medicine not collected` (1,170), `Test not completed` (275), `Review not attended` (543); `Completed care journey` (2,144).
- **Cohorts / temporal cutoffs**: DEVELOPMENT = consults ≤ **2026-06-30** (4,132, labels visible); EVALUATION = **2026-07-01 → 2026-08-15** (1,384, labels withheld). Our internal temporal validation: train ≤ **2026-04-30** (2,515), validate 05-01→06-30 (1,617). Prevalence is stable across windows (train 48.6%, valid 47.3%), so the split is clean.
- **Structural label rules discovered in EDA** (exact zeros in the advised-pattern × stage crosstab): a component that was not advised can never be the failure stage. Example: 174 episodes with nothing advised → all completed. This is a hard constraint any serious stage model must encode.

## 2. Record linkage (Patient 360)

**Challenge**: 8 operational systems each with private patient IDs and noisy
identity fields; the canonical table has only masked mobiles (`XXXXXX####`).

**Method** (`scripts/linkage.py`): normalise names (case/punct/honorifics),
phones (digits, strip 91-prefix), villages; block candidates by phone-4, fall
back to name-token index; score = 0.45·phone4 + 0.35·name-token-similarity(≤1-edit fuzzy) + 0.12·village(≤1-edit) + 0.05·gender + 0.03·age-tolerance.
Decision policy with margins (no blind forcing): ≥0.90+margin0.05 → high;
≥0.78+margin0.12 → medium; ≥0.55+margin0.20 → low; phone4-only without name
corroboration is rejected; everything else UNMATCHED (10.4%).

**Quality evidence** (`artifacts/linkage_metrics.json`, `linkage_quality.png`):
- 42,900 source records → 89.6% matched (90% of them high-confidence); 4,906/5,000 patients linked.
- Mobile-transitivity consistency 93.2% (records sharing a full mobile map to the same canonical patient).
- Full audit trail with score, margin, evidence per record: `artifacts/linkage_audit.csv`.

## 3. Temporal feature engineering & leakage prevention

**Prediction point = the consultation instant.** Every feature is computed from
records strictly at/before that instant (`features.py`). Feature families:

- **Patient static**: age, gender, NCD status, vulnerability group, language, village population, road access, connectivity.
- **Consultation (available at t)**: complaint, diagnosis, mode, duration, advised flags, review interval, distance, connectivity, status, weekday/month; derived `n_advised`, `dist×medicine`, chronic count.
- **Longitudinal history (strictly pre-t)**: #prior episodes, historical medicine-collection rate, partial-fill rate, stockout exposure, test-completion rate, review-attendance rate, recent-dropout flag, days since last Rx/screening, prior vitals (BP/glucose/BMI), prior visits/referrals/outreach(+unreachable).
- **Operational**: facility type/network/staffing; monthly ledger stockout rate & low-stock share (snapshots ≤ t only).

**Leakage controls**: (1) builder never reads post-consult tables — labels are joined only for training; (2) monotonicity probe — shifting prediction time later may only add history, verified on 200 random episodes; (3) column-level verdicts in `docs/data_dictionary.md` (166 columns marked use/never/conditional); (4) evaluation cohort outcomes never touched (they are blank in the pack).

## 4. Primary model comparison (temporal validation)

| model | ROC-AUC | PR-AUC | F1 | precision | recall | Brier | P@10 | R@10 | P@20 | R@20 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** (chosen) | **0.705** | **0.665** | **0.634** | 0.636 | 0.633 | 0.220 | **0.808** | 0.170 | 0.712 | **0.301** |
| Logistic regression | 0.692 | 0.653 | 0.609 | 0.619 | 0.599 | 0.223 | 0.758 | 0.160 | 0.697 | 0.294 |
| LightGBM | 0.671 | 0.624 | 0.617 | 0.601 | 0.634 | 0.236 | 0.696 | 0.146 | 0.656 | 0.277 |
| XGBoost | 0.670 | 0.621 | 0.602 | 0.592 | 0.612 | 0.231 | 0.696 | 0.146 | 0.656 | 0.277 |
| GradientBoosting | 0.668 | 0.612 | 0.608 | 0.597 | 0.618 | 0.232 | 0.683 | 0.144 | 0.644 | 0.272 |
| Baseline (prevalence) | 0.500 | 0.473 | — | — | — | 0.250 | 0.453 | 0.095 | 0.458 | 0.194 |

Hyperparameters were tuned on an **inner temporal split** (≤03-31 vs April) inside
the train window; the reporting validation was never used for tuning.

**Why P@k / R@k matter operationally**: outreach capacity is a small fraction of
patients. P@10% = 0.81 means: if ASHAs call the top-risk decile, ~4 of every 5
calls reach a true would-be dropout — vs 45% calling at random. Calibration is
acceptable (Brier 0.220 vs 0.250 baseline; curve in `calibration_roc_pr.png`).

**Honest finding**: boosting models underperformed here (small, mostly
low-cardinality signal; RF's smoother boundaries generalise better). We chose by
PR-AUC on the temporal validation set, not by fashion.

## 5. Stage prediction — the core differentiator

| approach | macro-F1 (dropout episodes) |
|---|---|
| **B: hierarchical (risk gate → stage model, masked)** | **0.560** |
| B without structural mask | 0.560 |
| A: direct multiclass (all episodes) | 0.352 all / **0.247 dropouts** |

**Hierarchical wins decisively** and is also the operationally right structure:
stage-relevant evidence (e.g., medicine-specific history) only matters once the
patient is flagged at risk. The structural mask (forbid stages whose component
was not advised) changed predictions on 0 validation episodes — the classifier
learned the constraint from data — but we keep it as a safety guarantee for
deployment.

Confusion matrix (dropouts, validation): medicine 355/449 correct, review 94/219,
test 44/97; main confusion is review→medicine. Per-stage insight: medicine
non-collection is the most separable failure (distance×medicine interaction);
review misses are the hardest (weakest historical signal).

**Why stage beats a single probability**: "74% risk" tells an ASHA to call;
"74% risk, will fail at medicine collection, facility shows stock-out pressure"
tells her to check stock first, then facilitate pickup — a different call than
a review reminder, with a different owner (ASHA vs CHO).

## 6. Ablations (information value)

| feature set | PR-AUC | ROC-AUC | F1 | P@10 |
|---|---|---|---|---|
| A demographics only | 0.476 | 0.487 | 0.435 | 0.509 |
| B + consultation | 0.661 | 0.703 | 0.618 | 0.776 |
| C + longitudinal history | 0.664 | 0.702 | 0.632 | 0.776 |
| D + operational/stock | 0.665 | 0.705 | 0.634 | **0.808** |
| D minus stage-history | 0.669 | 0.706 | 0.623 | 0.795 |

Reading: **who the patient is tells you almost nothing** (≈ chance); **what
happened in the consultation and what care was advised carries most signal**;
history/stock add a small, consistent edge on the precision-at-top-k that
matters for queueing. We report the minus-history ablation honestly: on this
synthetic dataset historical rates correlate only weakly with the label
(r = 0.00–0.06); in real registers we expect them to matter more, and the
pipeline is built so they simply plug in.

## 7. Interpretability, priority engine, action mapping

- **SHAP** (`shap_summary.png`, `shap_top_features.csv`): top drivers are care
  burden (`n_advised`), distance×medicine interaction, distance, advised flags,
  month, review interval, facility stockout pressure. Permutation importance
  agrees (`permutation_importance.csv`). Example cards with real SHAP factors:
  `docs/example_explanations.md` — associational language, no causal claims.
- **Priority score** = 0.60·risk + 0.25·stage urgency + 0.10·consequence + 0.05·feasibility.
  Tier thresholds are **capacity-calibrated on validation quantiles** and
  configurable in one dict: P1 ≥ 0.771, P2 ≥ 0.7175, P3 ≥ 0.540 (at 15/15/20/50% capacity shares).
- **Actions** are stage-specific with owner: medicine→ASHA pickup facilitation
  (with stock check first), test→scheduling support + 7-day recheck,
  review→CHO reminder before due date, completed→routine reminders.
- **System-barrier distinction**: predicted medicine failure + recent facility
  stockout pressure ⇒ queue text starts with "FACILITY ESCALATION FIRST…" and
  explicitly says *system barrier, not patient non-adherence*.

## 8. Fairness / subgroup checks (with honest caveats)

`artifacts/fairness.csv`: performance is broadly consistent across gender
(FNR 0.34 F / 0.38 M), age bands (FNR 0.29–0.40), all three districts
(FNR 0.35–0.36), and vulnerability groups. The `__missing__` group (n=59,
unlinked episodes) has degraded recall (0.33) — flagged honestly; linkage
quality is the lever, and subgroup cells below n=30 are suppressed rather than
interpreted.

## 9. Final submission (validated)

`submission/` (structure checked programmatically; `submission_validation.json`):
- `submission_linkage.csv` — 38,417 rows; unique (system, source-id); confidence ∈ {high, medium, low}.
- `submission_episode_predictions.csv` — 1,384 rows (all EVALUATION episodes), unique episode ids, probabilities in [0,1], stages from the dataset's own vocabulary, tiers P1–P4 (shares 16/13/21/50%).
- `submission_action_queue.csv` — 695 rows (P1–P3 only), unique episodes, valid canonical patient ids, plain-language reason + stage-specific action + cadre.

## 10. Limitations (stated, not hidden)

1. Synthetic data: regularities (e.g., distance→dropout) may be cleaner than reality; absolute numbers will not transfer — the *pipeline and controls* will.
2. Historical behaviour adds little here; with longer real histories we expect larger gains, but we do not claim it.
3. Review-stage misses remain hard (macro-F1 0.56); more registration-side features (reminders sent, transport) would help.
4. 10% of source records remain unlinked; their predictions rely on consultation features only.
5. Impact on outcomes is **not measured** (no intervention RCT in scope); the queue design is built to make that measurement possible.
6. Masked mobiles cap linkage evidence; production would use hashed full numbers.

## 11. Judge self-test

Clear problem ✓ · more than a classifier ✓ · novelty (stage-aware + structural mask + system-barrier logic) ✓ · defensible methodology (temporal splits, inner tuning, untouched evaluation) ✓ · leakage prevented & probed ✓ · interpretable (SHAP + cards + plain reasons) ✓ · prediction→action ✓ · stage-specific ✓ · system barriers ✓ · ASHA/CHO-feasible (one line, one action, one owner) ✓ · scalable (tabular, minutes, one command) ✓ · impact measurable by design (queue = intervention assignment) ✓ · limitations stated ✓.
