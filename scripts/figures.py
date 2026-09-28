"""EDA figures for the case-a-thon pack (all artifacts under artifacts/figures)."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ART, FIG, DATA, STAGES

warnings.filterwarnings("ignore")

F = pd.read_csv(os.path.join(ART, "features.csv"))
F["consult_date"] = pd.to_datetime(F["consult_date"])
aud = pd.read_csv(os.path.join(ART, "linkage_audit.csv"))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# 1. episodes per week by cohort (temporal design visible)
wk = F.groupby([F.consult_date.dt.to_period("W"), "cohort"]).size().unstack()
wk.index = wk.index.to_timestamp()
wk.plot(kind="bar", stacked=True, ax=ax[0, 0], width=1.0,
        color=["#2c7fb8", "#f4a261"])
ax[0, 0].set_title("Episodes per week by cohort (temporal split design)")
ax[0, 0].set_xlabel("")
ax[0, 0].tick_params(axis="x", labelsize=6, rotation=90)
ax[0, 0].axvline(list(wk.index).index(wk.index[wk.index <= pd.Timestamp("2026-06-30")][-1]) + 0.5,
                 color="k", ls="--", lw=1)

# 2. dropout stage distribution (development)
dev = F[F.cohort == "DEVELOPMENT"]
cnt = dev.dropout_stage_label.value_counts().reindex(STAGES)
ax[0, 1].barh(range(len(cnt)), cnt.values, color=["#e76f51", "#e9c46a", "#2a9d8f", "#264653"])
ax[0, 1].set_yticks(range(len(cnt)))
ax[0, 1].set_yticklabels([s.replace(" not ", "\nnot ") for s in cnt.index], fontsize=9)
ax[0, 1].set_title("Outcome stages - DEVELOPMENT (labels visible)")
for i, v in enumerate(cnt.values):
    ax[0, 1].text(v + 10, i, str(int(v)), va="center", fontsize=9)

# 3. advised-pattern vs first failure stage (structural rules)
ct = pd.crosstab(dev.medicine_advised.astype(str) + dev.test_advised.astype(str)
                 + dev.review_advised.astype(str), dev.dropout_stage_label)
ct = ct[STAGES]
ct.plot(kind="bar", stacked=True, ax=ax[1, 0],
        color=["#e76f51", "#e9c46a", "#2a9d8f", "#264653"])
ax[1, 0].set_title("Care components advised vs outcome stage (structural mask)")
ax[1, 0].set_xlabel("medicine+test+review advised (bit pattern)")
ax[1, 0].tick_params(axis="x", rotation=0)

# 4. dropout rate vs distance (binned)
d = dev.dropna(subset=["distance_km"]).copy()
d["dbin"] = pd.cut(d["distance_km"], [0, 5, 10, 15, 20, 25, 30, 40])
gb = d.groupby("dbin")["lost_to_followup_label"].agg(["mean", "size"])
ax[1, 1].errorbar(range(len(gb)), gb["mean"], yerr=1.96 * np.sqrt(
    gb["mean"] * (1 - gb["mean"]) / gb["size"]), marker="o", capsize=3, color="#e76f51")
ax[1, 1].set_xticks(range(len(gb)))
ax[1, 1].set_xticklabels([str(i) for i in gb.index], fontsize=8, rotation=30)
ax[1, 1].set_ylabel("dropout rate")
ax[1, 1].set_title("Dropout rate vs distance to facility (95% CI)")
ax[1, 1].axhline(dev.lost_to_followup_label.mean(), ls=":", color="gray")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "eda_overview.png"), dpi=150)
print("saved", os.path.join(FIG, "eda_overview.png"))

# linkage quality figure
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.8))
pt = aud.groupby("source_system").apply(
    lambda g: pd.Series({"match_rate": g.matched.mean(),
                         "high_share": (g.confidence == "high").sum() / max(g.matched.sum(), 1)}))
pt.plot(kind="bar", ax=ax2[0], color=["#2c7fb8", "#f4a261"])
ax2[0].set_title("Linkage match rate / high-confidence share by system")
ax2[0].tick_params(axis="x", rotation=30)
ax2[0].set_ylim(0, 1.05)
ax2[1].hist(aud.match_score, bins=40, color="#264653")
ax2[1].axvline(0.90, color="r", ls="--", lw=1, label="auto-accept")
ax2[1].set_title("Match-score distribution (all candidates)")
ax2[1].legend()
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "linkage_quality.png"), dpi=150)
print("saved", os.path.join(FIG, "linkage_quality.png"))
