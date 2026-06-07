#!/usr/bin/env python3
"""Plot training curves comparing v2/v3/v4 for C and G models."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LOGS = Path("/work/jl1401/icl_protein_disease/logs")
OUT  = Path("/work/jl1401/icl_protein_disease/results")

def parse_log(path):
    train_loss, val_loss, val_auroc = [], [], []
    with open(path) as f:
        for line in f:
            m = re.match(r'\s+Train Loss:\s+([\d.]+)', line)
            if m:
                train_loss.append(float(m.group(1)))
            m = re.match(r'\s+Val Loss:\s+([\d.]+)', line)
            if m:
                val_loss.append(float(m.group(1)))
            m = re.search(r'Val AUROC:\s+([\d.]+)', line)
            if m:
                val_auroc.append(float(m.group(1)))
    return train_loss, val_loss, val_auroc

configs = {
    "C": {
        "v2": LOGS / "train_c_v2_46548234.log",
        "v3": LOGS / "train_c_v3_46818057.log",
        "v4": LOGS / "train_c_v4_47009898.log",
    },
    "G": {
        "v2": LOGS / "train_g_v2_46548235.log",
        "v3": LOGS / "train_g_v3_46818058.log",
        "v4": LOGS / "train_g_v4_47009899.log",
    },
}

colors = {"v2": "#4C72B0", "v3": "#DD8452", "v4": "#55A868"}
styles = {"v2": "--", "v3": "-.", "v4": "-"}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Training Curves: v2 / v3 / v4  (C model top row, G model bottom row)",
             fontsize=13, fontweight="bold")

# Columns: Train Loss | Val Loss | Val AUROC
col_titles = ["Train Loss (BCE)", "Val Loss (BCE)", "Val AUROC (val set)"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11)

for row, block in enumerate(["C", "G"]):
    axes[row, 0].set_ylabel(f"{block} Model", fontsize=12, fontweight="bold")

    ax_tl  = axes[row, 0]
    ax_vl  = axes[row, 1]
    ax_auc = axes[row, 2]

    for ver, log_path in configs[block].items():
        train_loss, val_loss, val_auroc = parse_log(log_path)
        ep_tl  = np.arange(1, len(train_loss) + 1)
        ep_vl  = np.arange(1, len(val_loss)   + 1)
        ep_auc = np.arange(1, len(val_auroc)  + 1)

        label = f"{ver} ({len(train_loss)} ep)"
        kw = dict(color=colors[ver], linestyle=styles[ver], linewidth=2.0, label=label, alpha=0.9)

        ax_tl.plot(ep_tl, train_loss, **kw)
        ax_vl.plot(ep_vl, val_loss,   **kw)
        if val_auroc:
            ax_auc.plot(ep_auc, val_auroc, **kw)

        best_auc = f", best AUROC={max(val_auroc):.4f}" if val_auroc else ""
        print(f"{block} {ver}: {len(train_loss)} epochs | "
              f"train_loss {train_loss[0]:.3f}→{train_loss[-1]:.3f} | "
              f"val_loss {val_loss[0]:.3f}→{val_loss[-1]:.3f}{best_auc}")

    for ax in [ax_tl, ax_vl, ax_auc]:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # Cap val loss at 3 for readability (v2/v3 blow up)
    ax_vl.set_ylim(0, min(ax_vl.get_ylim()[1], 12))

    # Mark best v4 val AUROC
    _, _, auc_v4 = parse_log(configs[block]["v4"])
    if auc_v4:
        best_ep = int(np.argmax(auc_v4)) + 1
        ax_auc.axvline(best_ep, color=colors["v4"], linestyle=":", alpha=0.6)
        ax_auc.annotate(f"best={max(auc_v4):.4f}\nep{best_ep}",
                        xy=(best_ep, max(auc_v4)), xytext=(best_ep + 1, max(auc_v4) - 0.02),
                        fontsize=8, color=colors["v4"])
    ax_auc.set_ylim(0.55, 0.92)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = OUT / "v4_training_curves_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
