"""
Line 2 - K scaling eval: test v5 G model at K=64, 128, 256.
K=64 baseline reused from existing summary_G.json where available.
K=128 and K=256 evaluated fresh.

Key efficiency fix: use ONE context per seed (sent to GPU once),
then batch all queries against that fixed context.
This reduces GPU memory transfers from O(N/batch) to O(1) per seed.

Output: analysis_results/k_scaling/k_scaling_g.json
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT

PROJECT  = '/work/jl1401/icl_protein_disease'
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT      = os.path.join(PROJECT, 'analysis_results/k_scaling')
os.makedirs(OUT, exist_ok=True)

MODEL_G   = os.path.join(PROJECT, 'checkpoints_g_model_v5', 'best_model.pt')
SUMMARY_G = os.path.join(PROJECT, 'analysis_results/expanded_validation/summary_G.json')

N_SEEDS    = 15
K_VALUES   = [64, 128, 256]   # K=64 loaded from cache where possible
N_POS_MIN  = 150
BATCH_QRY  = 512   # larger query batch (no context per-batch overhead)
N_EVAL_MAX = 10000  # subsample queries to cap eval time (47 diseases × ~5 min each)

TRAIN_G = {'G439', 'G43', 'G56', 'G560'}


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


def icl_eval_one_k(model, X, y, device, ctx_size, n_seeds=N_SEEDS, batch_qry=BATCH_QRY,
                   n_eval_max=N_EVAL_MAX):
    """
    Efficient ICL eval: sample ONE context per seed, batch ALL queries against it.
    GPU memory transfer: 1 context per seed (not 1 per batch).
    Queries subsampled to n_eval_max to keep runtime bounded.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    pos_idx = np.where(y.astype(int) == 1)[0]
    neg_idx = np.where(y.astype(int) == 0)[0]
    N = len(y)

    # Subsample queries if dataset is large (caps per-disease eval time)
    if N > n_eval_max:
        rng_sub = np.random.RandomState(0)
        eval_idx = rng_sub.choice(N, n_eval_max, replace=False)
        eval_idx.sort()
        X = X[eval_idx]
        y = y[eval_idx]
        pos_idx = np.where(y.astype(int) == 1)[0]
        neg_idx = np.where(y.astype(int) == 0)[0]
        N = len(y)

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return float('nan'), []

    prevalence = len(pos_idx) / max(N, 1)
    target_ratio = min(prevalence * 3, 0.3)
    n_ctx_pos = max(1, int(ctx_size * target_ratio))
    n_ctx_neg = max(1, ctx_size - n_ctx_pos)

    # Check we have enough samples to fill the context
    if len(pos_idx) < 1 or len(neg_idx) < 1:
        return float('nan'), []

    X_t = torch.from_numpy(X).to(device)   # (N, D) — queries stay on GPU

    seed_aurocs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)

        # Sample ONE context for this seed
        ctx_p = pos_idx[rng.choice(len(pos_idx), n_ctx_pos, replace=(len(pos_idx) < n_ctx_pos))]
        ctx_n = neg_idx[rng.choice(len(neg_idx), n_ctx_neg, replace=(len(neg_idx) < n_ctx_neg))]
        ctx_idx = np.concatenate([ctx_p, ctx_n])
        rng.shuffle(ctx_idx)

        ctx_X_t = torch.from_numpy(X[ctx_idx]).to(device)   # (K, D)
        ctx_y_t = torch.from_numpy(y[ctx_idx]).to(device)   # (K,)

        # Apply fixed context to ALL queries in batches
        all_logits = []
        for start in range(0, N, batch_qry):
            end = min(start + batch_qry, N)
            bsz = end - start

            # Expand context to match batch size
            ctx_X_batch = ctx_X_t.unsqueeze(0).expand(bsz, -1, -1)   # (bsz, K, D)
            ctx_y_batch = ctx_y_t.unsqueeze(0).expand(bsz, -1).long()  # (bsz, K)
            qry_X_batch = X_t[start:end]                               # (bsz, D)

            # Make contiguous to avoid expand overhead in model
            ctx_X_batch = ctx_X_batch.contiguous()

            with torch.no_grad():
                logits = model({
                    'context_proteins': ctx_X_batch,
                    'context_labels':   ctx_y_batch,
                    'query_proteins':   qry_X_batch,
                }).cpu().numpy()
            all_logits.append(logits)

        logits_all = np.concatenate(all_logits)
        try:
            auc = float(roc_auc_score(y, logits_all))
        except Exception:
            auc = float('nan')
        seed_aurocs.append(auc)

    valid = [a for a in seed_aurocs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float('nan'), seed_aurocs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load K=64 cached results from expanded validation
    k64_cache = {}
    if os.path.exists(SUMMARY_G):
        with open(SUMMARY_G) as f:
            summ = json.load(f)
        for code, v in summ['per_disease'].items():
            k64_cache[code] = v['icl_abs']
        print(f"Loaded K=64 cache for {len(k64_cache)} diseases", flush=True)

    print("\nLoading v5 G model...", flush=True)
    ckpt = torch.load(MODEL_G, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model = InContextTransformerGPT(protein_dim=2941, hidden_dim=768, dropout=0.2)
    model.load_state_dict(state)
    model.to(device).eval()
    print("Model loaded.", flush=True)

    print("\nLoading protein data...", flush=True)
    protein_df = load_protein_data()
    print("\nLoading binary CSV...", flush=True)
    disease_df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'), compression='gzip')
    print(f"Binary CSV: {disease_df.shape}", flush=True)

    # Build evaluation pool
    # Pre-filter by positive count BEFORE expensive merge (250 cols → ~47)
    print("\nBuilding eval pool...", flush=True)
    g_cols = [(col, col.split('#')[1]) for col in disease_df.columns
              if col.startswith('Union#G')]
    print(f"  Found {len(g_cols)} G-block columns to screen", flush=True)
    pool = {}
    for col, code in g_cols:
        if code in TRAIN_G:
            continue
        if disease_df[col].sum() < N_POS_MIN:   # pandas sum ignores NaN; avoids merge
            continue
        label_df = disease_df[['userID', col]].dropna()
        merged = protein_df.merge(label_df, on='userID', how='inner')
        y = merged[col].values.astype(np.float32)
        X = merged.drop(columns=['userID', col]).values.astype(np.float32)
        if not np.isfinite(X).all():
            X[~np.isfinite(X)] = 0.0
        pool[code] = (X, y)
        print(f"  {code}: N={len(y)}, N+={int(y.sum())}", flush=True)
    print(f"Eval pool: {len(pool)} diseases", flush=True)

    # Load partial results
    resume_path = os.path.join(OUT, 'k_scaling_g.json')
    results = {}
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} diseases already done", flush=True)

    for code in sorted(pool.keys()):
        # Check if all K values done for this disease
        if code in results:
            row = results[code]
            all_done = all(f'k{K}_mean' in row for K in K_VALUES)
            # Also check K=64 cache
            if 64 not in K_VALUES and code in k64_cache:
                row['k64_mean'] = k64_cache[code]
            if all_done:
                print(f"  {code}: already done, skipping", flush=True)
                continue

        X, y = pool[code]
        n_pos = int(y.sum())
        print(f"\n--- {code}: N={len(y)}, N+={n_pos} ---", flush=True)

        if code not in results:
            results[code] = {'n_pos': n_pos}
        row = results[code]

        # K=64 from cache if available
        if 'k64_mean' not in row:
            if code in k64_cache:
                row['k64_mean'] = k64_cache[code]
                row['k64_source'] = 'cache'
                print(f"  K= 64: {k64_cache[code]:.4f} (cached)", flush=True)
            else:
                mean, seeds = icl_eval_one_k(model, X, y, device, ctx_size=64)
                ci95 = compute_ci95(seeds)
                row['k64_mean'] = mean
                row['k64_ci95'] = ci95
                row['k64_seeds'] = seeds
                row['k64_source'] = 'eval'
                print(f"  K= 64: {mean:.4f} ± {ci95:.4f}", flush=True)

        # Evaluate K=128 and K=256
        for K in [128, 256]:
            key = f'k{K}_mean'
            if key in row:
                print(f"  K={K:3d}: {row[key]:.4f} (done)", flush=True)
                continue
            if K > len(y) // 2:
                row[f'k{K}_mean'] = float('nan')
                print(f"  K={K:3d}: skipped (N too small)", flush=True)
                continue
            mean, seeds = icl_eval_one_k(model, X, y, device, ctx_size=K)
            ci95 = compute_ci95(seeds)
            row[f'k{K}_mean'] = mean
            row[f'k{K}_ci95'] = ci95
            row[f'k{K}_seeds'] = seeds
            print(f"  K={K:3d}: {mean:.4f} ± {ci95:.4f}", flush=True)

        results[code] = row
        with open(resume_path, 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*65}", flush=True)
    print(f"K SCALING SUMMARY (G block v5 model, {len(results)} diseases)", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"{'Code':<8}  {'K=64':>7}  {'K=128':>7}  {'K=256':>7}  {'128-64':>7}  {'256-64':>7}", flush=True)
    for code in sorted(results.keys()):
        v = results[code]
        k64 = v.get('k64_mean', float('nan'))
        k128 = v.get('k128_mean', float('nan'))
        k256 = v.get('k256_mean', float('nan'))
        d128 = k128 - k64 if not any(np.isnan([k128, k64])) else float('nan')
        d256 = k256 - k64 if not any(np.isnan([k256, k64])) else float('nan')
        print(f"{code:<8}  {k64:7.4f}  {k128:7.4f}  {k256:7.4f}  {d128:+7.4f}  {d256:+7.4f}", flush=True)

    # Mean Δ
    d128_vals = [v.get('k128_mean', float('nan')) - v.get('k64_mean', float('nan'))
                 for v in results.values()]
    d256_vals = [v.get('k256_mean', float('nan')) - v.get('k64_mean', float('nan'))
                 for v in results.values()]
    d128_clean = [x for x in d128_vals if not np.isnan(x)]
    d256_clean = [x for x in d256_vals if not np.isnan(x)]
    print(f"\nMean Δ K=128 vs K=64: {np.mean(d128_clean):+.4f} (n={len(d128_clean)})", flush=True)
    print(f"Mean Δ K=256 vs K=64: {np.mean(d256_clean):+.4f} (n={len(d256_clean)})", flush=True)
    print(f"\nResults: {resume_path}", flush=True)


def compute_ci95(seeds):
    valid = [s for s in seeds if not np.isnan(s)]
    if len(valid) < 2:
        return float('nan')
    return float(1.96 * np.std(valid) / np.sqrt(len(valid)))


if __name__ == '__main__':
    main()
