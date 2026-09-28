"""Generate example patient explanation cards (validation episodes) using SHAP
values from the chosen model + predicted stage + priority reasoning.
Writes docs/example_explanations.md."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ART, MODELS, ROOT, TRAIN_END, VALID_START, COMPLETED_STAGE
from train import add_derived, encode, _expand_sets, pipe, stage_mask

with open(os.path.join(MODELS, "dropout_model.pkl"), "rb") as f:
    dm = pickle.load(f)
with open(os.path.join(MODELS, "stage_model.pkl"), "rb") as f:
    sm = pickle.load(f)

F = pd.read_csv(os.path.join(ART, "features.csv"))
F = add_derived(F)
dev = F[(F.cohort == "DEVELOPMENT") & F.lost_to_followup_label.notna()].copy()
valid = dev[dev.consult_date >= pd.Timestamp(VALID_START)]
valid = valid[valid.dropout_stage_label.isin(
    ["Medicine not collected", "Test not completed", "Review not attended",
     COMPLETED_STAGE])]
valid["y"] = valid["lost_to_followup_label"].astype(int)

cols = dm["columns"]
X = encode(valid, _expand_sets()["D_all"]).reindex(columns=cols, fill_value=0.0)
model = dm["pipeline"]
p = model.predict_proba(X.values)[:, 1]

# SHAP for the chosen model
expl = shap.TreeExplainer(model.named_steps["clf"])
Xt = model.named_steps["imp"].transform(X.values)
sv = expl.shap_values(Xt)
if isinstance(sv, list):
    sv = sv[-1]
elif isinstance(sv, np.ndarray) and sv.ndim == 3:
    sv = sv[:, :, -1]

# stage predictions for the high-risk half
stage_cols = sm["columns"]
classes = sm["classes"]
from config import ID2STAGE
class_names = np.array([ID2STAGE[int(c)] for c in classes])
Xs = encode(valid, _expand_sets()["D_all"]).reindex(columns=stage_cols, fill_value=0.0)
sp = sm["pipeline"].predict_proba(Xs.values)
sp = np.where(stage_mask(valid, classes), sp, -np.inf)
stg = class_names[sp.argmax(1)]
thr = float(np.median(p))
stg_final = np.where(p >= thr, stg, COMPLETED_STAGE)

# top contributing factors per patient (signed)
feat_names = list(X.columns)
cards = []
order = np.argsort(-p)
seen_stage = set()
pick = []
for i in order:
    s = stg_final[i]
    key = s if valid["y"].values[i] == 1 else "completed"
    if key not in seen_stage:
        seen_stage.add(key)
        pick.append(i)
    if len(pick) >= 6:
        break

for i in pick:
    svals = sv[i]
    top = np.argsort(-np.abs(svals))[:4]
    factors = []
    for j in top:
        name = feat_names[j]
        val = X.values[i, j]
        direction = "higher" if svals[j] > 0 else "lower"
        factors.append(f"`{name}`={val:.2f} (pushes risk {direction})")
    r = valid.iloc[i]
    true_stage = r["dropout_stage_label"]
    outcome = ("correct" if (r["y"] == 1) == (p[i] >= 0.5) else
               ("missed dropout (false negative)" if r["y"] == 1 else "false alarm"))
    stage_note = ""
    if r["y"] == 1:
        stage_note = f" - {'MATCH' if stg_final[i] == true_stage else 'miss'} vs true stage '{true_stage}'"
    cards.append(f"""### Episode {r['episode_id']} (patient {r['patient_id'] or 'unlinked'})
- Predicted risk: **{p[i]*100:.0f}%** ({'dropout' if p[i]>=0.5 else 'likely complete'})
- Likely failure stage: **{stg_final[i]}**{stage_note}
- Associated factors (SHAP, not causal): {'; '.join(factors)}
- Validated outcome: actual={'dropout' if r['y']==1 else 'completed'} -> {outcome}
""")

os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
with open(os.path.join(ROOT, "docs", "example_explanations.md"), "w") as f:
    f.write("# Example patient explanations (temporal validation episodes)\n\n")
    f.write("Language is deliberately associational: factors are the ones the model "
            "weighed most for this episode; they are not causal claims.\n\n")
    f.write("\n".join(cards))
print("wrote cards for", len(pick), "episodes")
print("\n".join(cards[:2]))
