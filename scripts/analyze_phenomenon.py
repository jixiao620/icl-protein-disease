"""
A-D analysis: explain why cross-block ICL wins on C18/C43/G35 but loses on others.
A: Define Δ = ICL_mean(v2/v3/v5/v7) - max(XGB, DNN); scatter plots
B: Protein transfer distance via Cohen's d cosine
C: Linear separability (LR vs XGB gap in 5-fold CV)
D: Joint Spearman attribution table
Outputs → analysis_results/
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT = '/work/jl1401/icl_protein_disease'
OUT     = os.path.join(PROJECT, 'analysis_results')
os.makedirs(OUT, exist_ok=True)

# ─── Ground-truth numbers from spec ─────────────────────────────────────────
DISEASES_C = ['C34', 'C18', 'C43', 'C53', 'C54']
DISEASES_G = ['G40', 'G20', 'G30', 'G35']
ALL_DISEASES = DISEASES_C + DISEASES_G

TRAIN_C = ['C44', 'C50', 'C61', 'C809']
TRAIN_G = ['G439', 'G43', 'G56', 'G560']

N_POS = {
    'C34': 734, 'C18': 816, 'C43': 858, 'C53': 445, 'C54': 300,
    'G40': 985, 'G20': 870, 'G30': 667, 'G35': 470,
}

XGB_AUROC = {
    'C34': 0.7951, 'C18': 0.6838, 'C43': 0.6263, 'C53': 0.7403, 'C54': 0.7794,
    'G40': 0.6952, 'G20': 0.7358, 'G30': 0.8460, 'G35': 0.6922,
}
DNN_AUROC = {
    'C34': 0.8211, 'C18': 0.6699, 'C43': 0.6114, 'C53': 0.7734, 'C54': 0.8226,
    'G40': 0.7142, 'G20': 0.7908, 'G30': 0.8481, 'G35': 0.7443,
}
SD_ICL = {
    'C34': 0.7251, 'C18': 0.6432, 'C43': 0.6083, 'C53': 0.7420, 'C54': 0.7336,
    'G40': 0.6880, 'G20': 0.8195, 'G30': 0.8301, 'G35': 0.7630,
}

# ─── Load ICL cross-block results (v2/v3/v5/v7) ─────────────────────────────
def load_auroc(path, disease_key=None):
    """Return dict disease→auroc from a result JSON."""
    with open(path) as f:
        d = json.load(f)
    if disease_key:
        d = d[disease_key]  # e.g. d['final_crossblock']
    return {k: v['auroc'] if isinstance(v, dict) else v
            for k, v in d.items() if isinstance(k, str) and k[0] in 'CG'}

R = os.path.join(PROJECT, 'results')
icl_versions = {
    'C': {
        'v2': load_auroc(f'{R}/c_v2_best_on_c.json'),
        'v3': load_auroc(f'{R}/c_v3_best_on_c.json'),
        'v5': load_auroc(f'{R}/c_v5_best_crossblock_ctx64.json'),
        'v7': load_auroc(f'{R}/processed_data_c_v6_v7_results.json',
                         disease_key='final_crossblock'),
    },
    'G': {
        'v2': load_auroc(f'{R}/g_v2_best_on_g.json'),
        'v3': load_auroc(f'{R}/g_v3_best_on_g.json'),
        'v4': load_auroc(f'{R}/g_v4_best_on_g.json'),
        'v5': load_auroc(f'{R}/g_v5_best_crossblock_ctx64.json'),
        'v7': load_auroc(f'{R}/processed_data_g_v6_v7_results.json',
                         disease_key='final_crossblock'),
    },
}

# Compute per-disease mean/std across available versions
icl_mean, icl_std = {}, {}
for dis in ALL_DISEASES:
    block = 'C' if dis.startswith('C') else 'G'
    vals = [v[dis] for v in icl_versions[block].values() if dis in v]
    icl_mean[dis] = float(np.mean(vals))
    icl_std[dis]  = float(np.std(vals))

baseline_max = {d: max(XGB_AUROC[d], DNN_AUROC[d]) for d in ALL_DISEASES}
delta = {d: icl_mean[d] - baseline_max[d] for d in ALL_DISEASES}

print("=" * 60)
print("A. Δ = ICL_mean - max(XGB, DNN)")
print("=" * 60)
for d in ALL_DISEASES:
    tag = "WIN" if delta[d] > 0 else "lose"
    print(f"  {d}: ICL={icl_mean[d]:.4f}±{icl_std[d]:.4f}  "
          f"base={baseline_max[d]:.4f}  Δ={delta[d]:+.4f}  {tag}")

# ─── A: Scatter plots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

colors = ['#e74c3c' if delta[d] > 0 else '#3498db' for d in ALL_DISEASES]

ax = axes[0]
ax.scatter([icl_mean[d] for d in ALL_DISEASES],
           [baseline_max[d] for d in ALL_DISEASES], c=colors, s=80, zorder=3)
for d in ALL_DISEASES:
    ax.annotate(d, (icl_mean[d], baseline_max[d]),
                textcoords='offset points', xytext=(4, 3), fontsize=7)
mn = min(min(icl_mean.values()), min(baseline_max.values())) - 0.02
mx = max(max(icl_mean.values()), max(baseline_max.values())) + 0.02
ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('Cross-block ICL AUROC (mean v2/v3/v5/v7)')
ax.set_ylabel('max(XGB, DNN) AUROC')
ax.set_title('A: ICL vs Baseline (red=ICL wins)')
ax.grid(alpha=0.3)

ax = axes[1]
npos_vals = [N_POS[d] for d in ALL_DISEASES]
delta_vals = [delta[d] for d in ALL_DISEASES]
ax.scatter(npos_vals, delta_vals, c=colors, s=80, zorder=3)
for d in ALL_DISEASES:
    ax.annotate(d, (N_POS[d], delta[d]),
                textcoords='offset points', xytext=(4, 3), fontsize=7)
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
r, p = stats.spearmanr(npos_vals, delta_vals)
ax.set_xlabel("N⁺ (positives)")
ax.set_ylabel("Δ = ICL - max(XGB,DNN)")
ax.set_title(f'A: Δ vs N⁺  (Spearman ρ={r:.2f}, p={p:.2f})')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT}/A_scatter.png', dpi=150)
plt.close()
print(f"\nSaved A_scatter.png  (Δ vs N⁺ Spearman ρ={r:.3f})")

# ─── B: Protein transfer distance (Cohen's d cosine) ────────────────────────
print("\n" + "=" * 60)
print("B. Cohen's d cosine: test disease vs training diseases")
print("=" * 60)

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

print("Loading pkl files...", flush=True)
data_c_train = load_pkl(f'{PROJECT}/processed_data_c/train_data.pkl')
data_c_test  = load_pkl(f'{PROJECT}/processed_data_c/test_data.pkl')
data_g_train = load_pkl(f'{PROJECT}/processed_data_g/train_data.pkl')
data_g_test  = load_pkl(f'{PROJECT}/processed_data_g/test_data.pkl')
print("Loaded.", flush=True)

def cohens_d_vector(X, y):
    """Per-protein Cohen's d: (mean_case - mean_ctrl) / pooled_std. Shape: (n_proteins,)"""
    pos = X[y == 1]
    neg = X[y == 0]
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    # pooled std
    n1, n2 = len(pos), len(neg)
    pooled_var = ((n1 - 1) * pos.var(axis=0) + (n2 - 1) * neg.var(axis=0)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    return mean_diff / pooled_std

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# Compute Cohen's d for all training and test diseases
all_data = {}
all_data.update(data_c_train)
all_data.update(data_c_test)
all_data.update(data_g_train)
all_data.update(data_g_test)

print("Computing Cohen's d vectors...", flush=True)
d_vecs = {}
for code, (X, y) in all_data.items():
    d_vecs[code] = cohens_d_vector(X, y.astype(int))
    print(f"  {code}: N={len(y)}, N+={int(y.sum())}, |d|={np.linalg.norm(d_vecs[code]):.2f}")

cosine_max, cosine_mean = {}, {}
for dis in ALL_DISEASES:
    block = 'C' if dis.startswith('C') else 'G'
    train_codes = TRAIN_C if block == 'C' else TRAIN_G
    sims = [cosine(d_vecs[dis], d_vecs[tr]) for tr in train_codes if tr in d_vecs]
    cosine_max[dis]  = float(np.max(sims))
    cosine_mean[dis] = float(np.mean(sims))
    print(f"  {dis}: cosine_max={cosine_max[dis]:.4f}  cosine_mean={cosine_mean[dis]:.4f}  "
          f"per-train: {dict(zip(train_codes, [f'{s:.3f}' for s in sims]))}")

# ─── C: Linear separability ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("C. Linear separability: 5-fold stratified CV (LR vs XGB)")
print("=" * 60)

lr_auroc, xgb_auroc_cv, nonlin_gap = {}, {}, {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for dis in ALL_DISEASES:
    X, y = all_data[dis]
    y = y.astype(int)
    print(f"\n  {dis} (N={len(y)}, N+={y.sum()})...", flush=True)

    # L2 Logistic Regression with StandardScaler
    lr_pipe = Pipeline([('scaler', StandardScaler()),
                        ('lr', LogisticRegression(C=1.0, max_iter=1000,
                                                   solver='lbfgs', random_state=42))])
    lr_scores = cross_val_score(lr_pipe, X, y, cv=skf,
                                scoring='roc_auc', n_jobs=-1)
    lr_mean = float(lr_scores.mean())
    lr_auroc[dis] = lr_mean
    print(f"    LR   AUROC: {lr_mean:.4f} ± {lr_scores.std():.4f}")

    # XGBoost
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.5,
                         use_label_encoder=False, eval_metric='logloss',
                         random_state=42, n_jobs=-1, verbosity=0,
                         scale_pos_weight=float((y == 0).sum()) / max(y.sum(), 1))
    xgb_scores = cross_val_score(xgb, X, y, cv=skf,
                                  scoring='roc_auc', n_jobs=1)
    xgb_mean = float(xgb_scores.mean())
    xgb_auroc_cv[dis] = xgb_mean
    nonlin_gap[dis] = xgb_mean - lr_mean
    print(f"    XGB  AUROC: {xgb_mean:.4f} ± {xgb_scores.std():.4f}")
    print(f"    gap (XGB-LR): {nonlin_gap[dis]:+.4f}")

# ─── D: Joint attribution ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("D. Spearman correlations (descriptive, N=9)")
print("=" * 60)

features = {
    'cosine_max':    [cosine_max[d]    for d in ALL_DISEASES],
    'cosine_mean':   [cosine_mean[d]   for d in ALL_DISEASES],
    'LR_auroc':      [lr_auroc[d]      for d in ALL_DISEASES],
    'nonlin_gap':    [nonlin_gap[d]    for d in ALL_DISEASES],
    'N_pos':         [N_POS[d]         for d in ALL_DISEASES],
}
targets = {
    'ICL_abs':  [icl_mean[d]   for d in ALL_DISEASES],
    'Delta':    [delta[d]      for d in ALL_DISEASES],
}

print(f"\n{'Feature':<15} {'ρ(ICL_abs)':>12} {'p':>8} {'ρ(Δ)':>10} {'p':>8}")
print("-" * 55)
spearman_results = {}
for feat, vals in features.items():
    r1, p1 = stats.spearmanr(vals, targets['ICL_abs'])
    r2, p2 = stats.spearmanr(vals, targets['Delta'])
    spearman_results[feat] = {'rho_icl': r1, 'p_icl': p1, 'rho_delta': r2, 'p_delta': p2}
    print(f"  {feat:<13} {r1:+.3f}   p={p1:.3f}   {r2:+.3f}  p={p2:.3f}")

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
rows = []
for d in ALL_DISEASES:
    block = 'C' if d.startswith('C') else 'G'
    n_ver = len(icl_versions[block])
    rows.append({
        'disease':      d,
        'block':        block,
        'ICL_abs':      round(icl_mean[d], 4),
        'ICL_std':      round(icl_std[d], 4),
        'n_versions':   n_ver,
        'XGB':          XGB_AUROC[d],
        'DNN':          DNN_AUROC[d],
        'baseline_max': round(baseline_max[d], 4),
        'Delta':        round(delta[d], 4),
        'SD_ICL':       SD_ICL[d],
        'N_pos':        N_POS[d],
        'cosine_max':   round(cosine_max[d], 4),
        'cosine_mean':  round(cosine_mean[d], 4),
        'LR_auroc':     round(lr_auroc[d], 4),
        'XGB_cv':       round(xgb_auroc_cv[d], 4),
        'nonlin_gap':   round(nonlin_gap[d], 4),
        'winner':       'YES' if delta[d] > 0 else 'no',
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(f'{OUT}/summary_table.csv', index=False)

# Save Spearman results
with open(f'{OUT}/spearman_results.json', 'w') as f:
    json.dump(spearman_results, f, indent=2)

# ─── B+C scatter ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#e74c3c' if delta[d] > 0 else '#3498db' for d in ALL_DISEASES]

def scatter_feat(ax, xkey, xlabel, ykey, ylabel, title):
    xv = [features[xkey][i] for i in range(len(ALL_DISEASES))]
    yv = targets[ykey]
    ax.scatter(xv, yv, c=colors, s=80, zorder=3)
    for i, d in enumerate(ALL_DISEASES):
        ax.annotate(d, (xv[i], yv[i]),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    r, p = stats.spearmanr(xv, yv)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nSpearman ρ={r:.2f}, p={p:.2f}')
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
    ax.grid(alpha=0.3)

scatter_feat(axes[0], 'cosine_max', 'cosine_max (transfer similarity)',
             'ICL_abs', 'ICL_abs AUROC', 'B: Transfer distance vs ICL abs')
scatter_feat(axes[1], 'nonlin_gap', 'nonlinearity gap (XGB_cv - LR_cv)',
             'ICL_abs', 'ICL_abs AUROC', 'C: Nonlinearity gap vs ICL abs')
scatter_feat(axes[2], 'cosine_max', 'cosine_max',
             'Delta', 'Δ = ICL - max(XGB,DNN)', 'B+A: Transfer distance vs Δ')

plt.tight_layout()
plt.savefig(f'{OUT}/BC_scatter.png', dpi=150)
plt.close()

print(f"\n=== DONE ===")
print(f"Outputs in {OUT}/:")
print(f"  A_scatter.png, BC_scatter.png, summary_table.csv, spearman_results.json")

# ─── Gate check ──────────────────────────────────────────────────────────────
winners  = [d for d in ALL_DISEASES if delta[d] > 0]
losers   = [d for d in ALL_DISEASES if delta[d] <= 0]
print(f"\n=== GATE CHECK ===")
print(f"Winners (Δ>0): {winners}")
print(f"Losers  (Δ≤0): {losers}")

win_cos  = np.mean([cosine_max[d] for d in winners])
lose_cos = np.mean([cosine_max[d] for d in losers])
win_gap  = np.mean([nonlin_gap[d] for d in winners])
lose_gap = np.mean([nonlin_gap[d] for d in losers])
print(f"\nMean cosine_max: winners={win_cos:.4f}  losers={lose_cos:.4f}  "
      f"direction={'OK (winners higher)' if win_cos > lose_cos else 'WRONG'}")
print(f"Mean nonlin_gap: winners={win_gap:.4f}  losers={lose_gap:.4f}  "
      f"direction={'OK (winners smaller gap)' if win_gap < lose_gap else 'WRONG'}")
gate = (win_cos > lose_cos) and (win_gap < lose_gap)
print(f"\nGATE: {'PASS → proceed to causal experiment' if gate else 'FAIL → revisit hypotheses'}")
