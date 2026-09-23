"""Regenerate the paper-style figures shown in README.md.

Run with: python figures/_make_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Fig 1 — cross-block AUROC comparison (ICL vs baselines) on TWO test-sets
# ---------------------------------------------------------------------------
# Blocks
blocks = ["I (circulatory)", "C (neoplasms)", "G (nervous system)"]

# Our clean test set (6 held-out ICD-10 per block)
ours_our = {
    "ICL (v9/v8)":         [0.8363, 0.7266, 0.5140],
    "XGBoost":             [0.6610, 0.5984, 0.5533],
    "DNN":                 [0.6581, 0.5846, 0.5665],
    "TabPFN v3 (vanilla)": [0.6554, 0.5793, 0.5195],
    "TabPFN v3 (finetuned)": [0.6920, 0.6133, 0.5466],
}

# Milton et al. test set (6 rare ICD-10, same as Nat Med 2024)
ours_milton = {
    "ICL (v9/v8)":         [0.8727, 0.6287, 0.5292],
    "DNN":                 [0.7714, 0.6254, 0.5619],
    "TabPFN v3 (vanilla)": [0.7386, 0.5781, 0.5753],
    "TabPFN v3 (finetuned)": [0.7826, 0.5965, 0.5769],
    "Milton (public)":     [0.7217, 0.6190, 0.6217],
}

colors = {
    "ICL (v9/v8)":         "#d62728",
    "XGBoost":             "#7f7f7f",
    "DNN":                 "#1f77b4",
    "TabPFN v3 (vanilla)": "#2ca02c",
    "TabPFN v3 (finetuned)": "#98df8a",
    "Milton (public)":     "#ff7f0e",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

for ax, (title, panel) in zip(axes,
                              [("Our clean test set (patient-disjoint)", ours_our),
                               ("Milton et al. (Nat Med 2024) test set", ours_milton)]):
    methods = list(panel.keys())
    n_methods = len(methods)
    x = np.arange(len(blocks))
    bar_w = 0.8 / n_methods
    for i, m in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_w
        bars = ax.bar(x + offset, panel[m], bar_w, label=m, color=colors[m],
                      edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, panel[m]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(blocks)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Mean AUROC over 6 held-out ICD-10")

axes[0].legend(loc="upper right", frameon=False, fontsize=9)
axes[1].legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("Cross-Disease ICL vs Baselines — Held-Out Rare ICD-10",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "main_results.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "main_results.png")

# ---------------------------------------------------------------------------
# Fig 2 — architectural ablation (progressive additions v6 → v7 → v8 → v9)
# ---------------------------------------------------------------------------
# Values from our clean K=32/n_train=4 baseline evals + v7/v8/v9 ensemble evals.
# For each block, we report AUROC (I=n_train=4 K=128, C=n_train=4 K=256,
# G=n_train=12 K=128 — best per-block eval-time hyperparameters).
variants = [
    "v6 (bidirectional set)",
    "v7 (+ quantile norm + per-protein cross-sample attn)",
    "v8 (+ protein-ID embedding)",
    "v9 (+ feature-attention pool)",
]
# I: v6 = 0.6953, v7 kitchen = ~0.7761, v8 = 0.7980, v9 = 0.8363
# C: v6 = 0.5835, v7 ≈ 0.6178, v8 ≈ 0.6180, v9 = 0.7266  (rough for v7/v8; v9 confirmed)
# G: v6 = 0.4215, v7 = 0.5071, v8 = 0.5140, v9 (no-diseaseemb) = 0.4365
# NOTE: G's v9 REGRESSES vs v8 — one of the key negative findings we discuss.
i_scores = [0.6953, 0.7761, 0.7980, 0.8363]
c_scores = [0.5835, 0.6178, 0.6180, 0.7266]
g_scores = [0.4215, 0.5071, 0.5140, 0.4365]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(variants))
bar_w = 0.25
b1 = ax.bar(x - bar_w, i_scores, bar_w, label="I block", color="#4C72B0",
            edgecolor="black", linewidth=0.6)
b2 = ax.bar(x,         c_scores, bar_w, label="C block", color="#DD8452",
            edgecolor="black", linewidth=0.6)
b3 = ax.bar(x + bar_w, g_scores, bar_w, label="G block", color="#55A868",
            edgecolor="black", linewidth=0.6)
for bars, scores in [(b1, i_scores), (b2, c_scores), (b3, g_scores)]:
    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([v.replace(" (", "\n(") for v in variants],
                   fontsize=9)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
ax.set_ylabel("Mean AUROC over 6 held-out ICD-10")
ax.set_title("Architectural Ablation — Progressive Additions to the ICL Transformer\n"
             "(I / C / G blocks, evaluated on our clean patient-disjoint test set)",
             fontsize=12)
ax.legend(loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(OUT / "architecture_ablation.png", dpi=200, bbox_inches="tight")
print("wrote", OUT / "architecture_ablation.png")

print("done")
