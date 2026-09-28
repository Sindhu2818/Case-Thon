# From Dropout Prediction to Care-Journey Assurance
*INFINUM 2026 — slide-by-slide narrative (12 slides)*

---

**Slide 1 — Problem**
Teleconsultation is counted as "completed", but care is not: medicines must be
collected, tests done, reviews attended. Follow-up data sits in 8 disconnected
registers. Patients don't "drop out" in general — **they fail at a specific
stage**, and today the system can't see which.

**Slide 2 — Insight (from the data)**
- 48% of development episodes are lost to follow-up.
- Failures concentrate in the first stage: **medicine 59% / review 27% / test 14%** of dropouts.
- The failure stage is *structurally* constrained by what was advised
  (nothing advised → nothing to fail). Any stage model must respect this.

**Slide 3 — Novel approach**
Two questions, one pipeline: **(1) will this episode fail? (2) where?**
Risk model → gated stage model → priority → stage-specific action.
Plus: a **system-barrier flag** so stock-out failures escalate the facility,
not the patient.

**Slide 4 — Data integration (Patient 360)**
42,900 records, 8 systems, no crosswalk, noisy names/phones/villages.
Graded-evidence fuzzy linkage (phone-suffix + name + geography + margins):
**89.6% matched, 90% high-confidence, 4,906/5,000 patients, 93% transitivity check**.
Every decision auditable.

**Slide 5 — Temporal discipline**
Prediction point = the consultation instant. Features = only what was known
then (history, consultation content, facility stock ≤ t). Temporal
train/valid split, inner split for tuning, evaluation cohort untouched.
Monotonicity probe passes. Column-level leakage dictionary for all 166 columns.

**Slide 6 — ML model (honest comparison)**
Random Forest wins on temporal validation — **PR-AUC 0.665, ROC 0.705,
P@10% = 0.81** (top-decile calls are 1.8× better than random outreach).
XGBoost/LightGBM/GradBoost tested and *beaten*; no deep learning needed.

**Slide 7 — Stage prediction (the differentiator)**
Hierarchical beats direct multiclass: **macro-F1 0.560 vs 0.247**.
Medicine failures are the most separable; review misses the hardest.

**Slide 8 — Why it matters (interpretability)**
Not "risk = 74%" but: *risk 74% → medicine collection — associated with long
distance (36 km), 3 advised components, facility stock pressure.*
SHAP-verified per-patient cards; associational language only.

**Slide 9 — Action, not alarm**
Each queue row = patient + risk + likely stage + P1–P4 + plain-language reason +
**one stage-specific action + one owner (ASHA/CHO)**. Medicine risk → facilitate
pickup, stock check first. Test risk → scheduling support. Review risk → CHO
reminder before due date. Priority tiers calibrated to outreach capacity.

**Slide 10 — System vs patient**
High risk + recent facility stock-outs ⇒ "**FACILITY ESCALATION FIRST** —
system barrier, not patient non-adherence." The queue protects patients from
being blamed for system gaps.

**Slide 11 — Measurable impact (by design)**
Queue = intervention assignment → ready for controlled rollout (outreach vs
usual care) to measure completed journeys, time-to-collection, stock-out
escalations. We claim prediction quality today; outcome impact only after measurement.

**Slide 12 — Reproducibility & governance**
One command reruns the entire solution end-to-end with identical numbers
(fixed seeds, temporal splits): `python run_pipeline.py`.
Every linkage decision is auditable, every queued action carries a plain-language
reason, and the evaluation cohort is never touched during development.
