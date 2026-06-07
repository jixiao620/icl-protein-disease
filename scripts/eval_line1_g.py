"""
Line 1 evaluation: test all 3 G pathway models on a common held-out test pool.

Hypothesis: each model should yield highest ICL AUROC on diseases from its own
pathway (not just beat baselines, but beat the OTHER pathway models too).

Output: analysis_results/line1_g/
  results.json  — per-disease AUROC for each model + baselines
  summary.txt   — formatted comparison table
  pathway_alignment.json — does the best model match the ICD pathway?
"""

import os, sys, json, pickle, random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT

PROJECT  = '/work/jl1401/icl_protein_disease'
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT      = os.path.join(PROJECT, 'analysis_results/line1_g')
os.makedirs(OUT, exist_ok=True)

N_SEEDS  = 15
CTX_SIZE = 64
N_POS_MIN = 150

# Training disease sets per pathway (excluded from test pool)
PATHWAY_TRAIN = {
    'pn': ['G54', 'G55', 'G57', 'G58'],
    'nd': ['G20', 'G30', 'G31', 'G35'],
    'hs': ['G44', 'G47', 'G50', 'G51'],
}
ALL_TRAIN = set(d for v in PATHWAY_TRAIN.values() for d in v)

# Subcodes of training diseases — exclude to avoid patient overlap
EXCLUDED_SUBCODES = {
    'G551', 'G552',   # subcodes of G55 (PN)
    'G576',           # subcode of G57 (PN)
    'G589',           # subcode of G58 (PN)
    'G473', 'G479',   # subcodes of G47 (HS)
    'G510',           # subcode of G51 (HS)
    'G309', 'G319',   # subcodes of G30/G31 (ND)
}

# Also exclude original v5 training diseases
V5_TRAIN = {'G439', 'G43', 'G56', 'G560'}

# ICD-based pathway assignment for test diseases (ground truth for analysis)
# None = ambiguous / no strong prior
PATHWAY_PRIOR = {
    # PN expected: polyneuropathies and peripheral nerve disorders
    'G62': 'pn', 'G629': 'pn', 'G63': 'pn', 'G632': 'pn', 'G64': 'pn',
    # ND expected: movement disorders, neurodegeneration
    'G12': 'nd', 'G122': 'nd', 'G24': 'nd', 'G25': 'nd',
    # HS expected: epilepsy (episodic), TIA
    'G40': 'hs', 'G409': 'hs', 'G45': 'hs', 'G459': 'hs',
}

MODEL_PATHS = {
    'pn': os.path.join(PROJECT, 'checkpoints_line1_g_pn', 'best_model.pt'),
    'nd': os.path.join(PROJECT, 'checkpoints_line1_g_nd', 'best_model.pt'),
    'hs': os.path.join(PROJECT, 'checkpoints_line1_g_hs', 'best_model.pt'),
}


def load_protein_data():
    fnames = sorted(f for f in os.listdir(DATA_DIR) if f.startswith('xa') and f.endswith('.gz'))
    dfs, cols = [], None
    for fn in fnames:
        fp = os.path.join(DATA_DIR, fn)
        if cols is None:
            df = pd.read_csv(fp, compression='gzip')
            cols = df.columns.tolist()
        else:
            df = pd.read_csv(fp, compression='gzip', header=None, names=cols)
        dfs.append(df)
        print(f"  {fn}: {len(df)} rows", flush=True)
    return pd.concat(dfs, ignore_index=True)


def find_col(disease_df, code):
    matches = [c for c in disease_df.columns if c.startswith(f'Union#{code}#')]
    if not matches:
        return None
    exact = [c for c in matches if c.split('#')[1] == code]
    return exact[0] if exact else matches[0]


def build_disease_pool(protein_df, disease_df):
    """Return dict of code -> (X, y) for all G diseases with N+ >= N_POS_MIN."""
    pool = {}
    for col in disease_df.columns:
        if not col.startswith('Union#G'):
            continue
        code = col.split('#')[1]
        if code in ALL_TRAIN or code in EXCLUDED_SUBCODES or code in V5_TRAIN:
            continue
        label_df = disease_df[['userID', col]].dropna()
        merged = protein_df.merge(label_df, on='userID', how='inner')
        y = merged[col].values.astype(np.float32)
        if int(y.sum()) < N_POS_MIN:
            continue
        X = merged.drop(columns=['userID', col]).values.astype(np.float32)
        if not np.isfinite(X).all():
            X[~np.isfinite(X)] = 0.0
        pool[code] = (X, y)
    return pool


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    # infer protein_dim from first projection layer
    model = InContextTransformerGPT(protein_dim=2941, hidden_dim=768, dropout=0.2)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def icl_eval_multiseed(model, X, y, device, n_seeds=N_SEEDS, ctx_size=CTX_SIZE, batch_size=128):
    """Run ICL evaluation with multiple seeds, return (mean_auroc, list_of_per_seed_aurocs)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    pos_idx = np.where(y.astype(int) == 1)[0]
    neg_idx = np.where(y.astype(int) == 0)[0]
    N = len(y)

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return float('nan'), []

    prevalence = len(pos_idx) / max(N, 1)
    target_ratio = min(prevalence * 3, 0.3)
    n_ctx_pos = int(ctx_size * target_ratio)
    n_ctx_neg = ctx_size - n_ctx_pos

    seed_aurocs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        all_logits = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bsz = end - start
            ctx_p = pos_idx[rng.choice(len(pos_idx), n_ctx_pos, replace=(len(pos_idx) < n_ctx_pos))]
            ctx_n = neg_idx[rng.choice(len(neg_idx), n_ctx_neg, replace=(len(neg_idx) < n_ctx_neg))]
            ctx_idx = np.concatenate([ctx_p, ctx_n])
            rng.shuffle(ctx_idx)

            ctx_X = torch.from_numpy(X[ctx_idx]).unsqueeze(0).expand(bsz, -1, -1).to(device)
            ctx_y = torch.from_numpy(y[ctx_idx]).unsqueeze(0).expand(bsz, -1).long().to(device)
            qry_X = torch.from_numpy(X[start:end]).to(device)

            with torch.no_grad():
                logits = model({
                    'context_proteins': ctx_X,
                    'context_labels':   ctx_y,
                    'query_proteins':   qry_X,
                }).cpu().numpy()
            all_logits.append(logits)

        logits_all = np.concatenate(all_logits)
        try:
            auc = float(roc_auc_score(y, logits_all))
        except Exception:
            auc = float('nan')
        seed_aurocs.append(auc)

    valid = [a for a in seed_aurocs if not np.isnan(a)]
    mean_auc = float(np.mean(valid)) if valid else float('nan')
    return mean_auc, seed_aurocs


def run_lr_xgb(X, y, n_splits=5):
    """5-fold LR + XGB, return (lr_auroc, xgb_auroc)."""
    if int(y.sum()) < n_splits or int((y == 0).sum()) < n_splits:
        return float('nan'), float('nan')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lr_pipe = Pipeline([('imp', SimpleImputer()), ('sc', StandardScaler()),
                        ('lr', LogisticRegression(max_iter=1000, C=0.1))])
    xgb_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='logloss',
                               verbosity=0, n_jobs=4)
    lr_auc  = float(np.mean(cross_val_score(lr_pipe, X, y, cv=skf, scoring='roc_auc')))
    xgb_auc = float(np.mean(cross_val_score(xgb_model, X, y, cv=skf, scoring='roc_auc')))
    return lr_auc, xgb_auc


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Check all model checkpoints exist
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"ERROR: checkpoint missing: {path}", flush=True)
            print("Train all 3 pathway models first.", flush=True)
            sys.exit(1)

    print("\nLoading protein data...", flush=True)
    protein_df = load_protein_data()
    print("\nLoading binary CSV...", flush=True)
    disease_df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'), compression='gzip')

    print("\nBuilding test pool (G diseases, N+ >= 150, excl. training + subcodes)...", flush=True)
    pool = build_disease_pool(protein_df, disease_df)
    print(f"Test pool: {len(pool)} diseases: {sorted(pool.keys())}", flush=True)

    print("\nLoading pathway models...", flush=True)
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}

    # Load cached LR/XGB from expanded validation if available
    cached_lr_xgb = {}
    summary_path = os.path.join(PROJECT, 'analysis_results/expanded_validation/summary_G.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summ = json.load(f)
        for code, v in summ['per_disease'].items():
            cached_lr_xgb[code] = (v['lr_auroc'], v['xgb_auroc'])
        print(f"Loaded cached LR/XGB for {len(cached_lr_xgb)} diseases", flush=True)

    results = {}
    resume_path = os.path.join(OUT, 'results.json')
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} diseases already done", flush=True)

    for code in sorted(pool.keys()):
        if code in results:
            print(f"  {code}: already done, skipping", flush=True)
            continue

        X, y = pool[code]
        n_pos = int(y.sum())
        print(f"\n--- {code}: N={len(y)}, N+={n_pos} ---", flush=True)

        row = {'n_pos': n_pos}

        # ICL eval for each pathway model
        for name, model in models.items():
            mean_auc, seeds = icl_eval_multiseed(model, X, y, device)
            row[f'icl_{name}'] = mean_auc
            row[f'icl_{name}_seeds'] = seeds
            print(f"  ICL-{name.upper()}: {mean_auc:.4f}", flush=True)

        # Baselines
        if code in cached_lr_xgb:
            lr_auc, xgb_auc = cached_lr_xgb[code]
        else:
            print(f"  Running LR+XGB baselines...", flush=True)
            lr_auc, xgb_auc = run_lr_xgb(X, y)
        row['lr_auroc']  = lr_auc
        row['xgb_auroc'] = xgb_auc
        print(f"  LR: {lr_auc:.4f}  XGB: {xgb_auc:.4f}", flush=True)

        # Best model
        icl_scores = {n: row[f'icl_{n}'] for n in models if not np.isnan(row[f'icl_{n}'])}
        best_model = max(icl_scores, key=icl_scores.get) if icl_scores else None
        row['best_icl_model'] = best_model
        row['pathway_prior'] = PATHWAY_PRIOR.get(code, None)
        row['prior_match'] = (best_model == PATHWAY_PRIOR.get(code)) if best_model and PATHWAY_PRIOR.get(code) else None

        results[code] = row
        with open(resume_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print_summary(results)
    save_summary(results)


def print_summary(results):
    print(f"\n{'='*80}", flush=True)
    print("LINE 1 G BLOCK RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Code':<8} {'Prior':>6}  {'PN':>6}  {'ND':>6}  {'HS':>6}  "
          f"{'LR':>6}  {'XGB':>6}  {'Best':>6}  {'Match':>6}", flush=True)

    match_count, total_prior = 0, 0
    for code in sorted(results.keys()):
        v = results[code]
        prior = v.get('pathway_prior', '-') or '-'
        best  = v.get('best_icl_model', '-') or '-'
        match = v.get('prior_match')
        match_str = 'Y' if match is True else ('N' if match is False else '-')
        if match is not None:
            total_prior += 1
            if match:
                match_count += 1
        pn  = v.get('icl_pn',  float('nan'))
        nd  = v.get('icl_nd',  float('nan'))
        hs  = v.get('icl_hs',  float('nan'))
        lr  = v.get('lr_auroc',  float('nan'))
        xgb = v.get('xgb_auroc', float('nan'))
        print(f"{code:<8} {prior:>6}  {pn:>6.4f}  {nd:>6.4f}  {hs:>6.4f}  "
              f"{lr:>6.4f}  {xgb:>6.4f}  {best:>6}   {match_str}", flush=True)

    if total_prior > 0:
        print(f"\nPathway alignment: {match_count}/{total_prior} diseases "
              f"({100*match_count/total_prior:.0f}%) best model matches ICD-based prior",
              flush=True)


def save_summary(results):
    # Pathway alignment analysis
    alignment = {'match': 0, 'mismatch': 0, 'no_prior': 0, 'per_pathway': {}}
    for pathway in ['pn', 'nd', 'hs']:
        alignment['per_pathway'][pathway] = {'correct': [], 'wrong': [], 'no_prior': []}

    for code, v in results.items():
        prior = v.get('pathway_prior')
        best  = v.get('best_icl_model')
        match = v.get('prior_match')
        if prior is None:
            alignment['no_prior'] += 1
            alignment['per_pathway'].get(best, {}).get('no_prior', []).append(code) if best else None
        elif match:
            alignment['match'] += 1
            alignment['per_pathway'][prior]['correct'].append(code)
        else:
            alignment['mismatch'] += 1
            alignment['per_pathway'][prior]['wrong'].append(code)

    out = os.path.join(OUT, 'pathway_alignment.json')
    with open(out, 'w') as f:
        json.dump(alignment, f, indent=2)

    # Per-model win counts vs baselines
    wins = {name: 0 for name in ['pn', 'nd', 'hs']}
    for v in results.values():
        for name in ['pn', 'nd', 'hs']:
            icl = v.get(f'icl_{name}', float('nan'))
            lr  = v.get('lr_auroc',  float('nan'))
            xgb = v.get('xgb_auroc', float('nan'))
            if not any(np.isnan([icl, lr, xgb])) and icl > lr and icl > xgb:
                wins[name] += 1

    print(f"\nBaseline wins (ICL > LR and ICL > XGB):", flush=True)
    for name, w in wins.items():
        print(f"  {name.upper()}: {w}/{len(results)}", flush=True)

    summary = {'alignment': alignment, 'baseline_wins': wins, 'n_diseases': len(results)}
    out = os.path.join(OUT, 'summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUT}/", flush=True)


if __name__ == '__main__':
    main()
