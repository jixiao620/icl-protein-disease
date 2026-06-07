"""
Block-level expanded validation — Steps 1-5.
Step 1: noise floor (15 seeds × existing test diseases)
Step 2: build candidate disease pool from binary_csv.gz
Step 3: compute cosine + LR baseline for each candidate
Step 4: ICL inference (15 seeds) on each candidate
Step 5: Spearman analysis + out-of-sample prediction
"""

import os, sys, json, pickle, random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT

PROJECT  = '/work/jl1401/icl_protein_disease'
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT      = os.path.join(PROJECT, 'analysis_results/expanded_validation')
os.makedirs(OUT, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_C  = ['C44', 'C50', 'C61', 'C809']
TRAIN_G  = ['G439', 'G43', 'G56', 'G560']
EXIST_C  = ['C34', 'C18', 'C43', 'C53', 'C54']
EXIST_G  = ['G40', 'G20', 'G30', 'G35']
N_SEEDS  = 15
N_POS_MIN = 150
CTX_SIZE  = 64

# 3-char prefixes of training diseases to exclude sub-codes
EXCL_PREFIX_C = {'C44', 'C50', 'C61', 'C80'}   # C809 → C80 prefix
EXCL_PREFIX_G = {'G43', 'G56'}                   # G439→G43, G560→G56, G560→G56


# ── Data loading ─────────────────────────────────────────────────────────────
def load_protein_data():
    print("Loading protein data...", flush=True)
    dfs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not (fname.startswith('xa') and fname.endswith('.gz')):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        if fname == 'xaa.gz':
            df = pd.read_csv(fpath, compression='gzip')
        else:
            df = pd.read_csv(fpath, compression='gzip', header=None,
                             names=pd.read_csv(os.path.join(DATA_DIR, 'xaa.gz'),
                                               compression='gzip', nrows=0).columns)
        dfs.append(df)
        print(f"  {fname}: {len(df)} rows", flush=True)
    protein_df = pd.concat(dfs, ignore_index=True)
    print(f"Total protein data: {protein_df.shape}", flush=True)
    return protein_df


def load_binary_csv():
    print("Loading binary disease labels...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'), compression='gzip')
    print(f"Binary CSV: {df.shape}", flush=True)
    return df


def build_disease_data(protein_df, binary_df, disease_code):
    col = f'HC_{disease_code}' if f'HC_{disease_code}' in binary_df.columns else disease_code
    if col not in binary_df.columns:
        return None, None
    label_df = binary_df[['userID', col]].dropna()
    merged = protein_df.merge(label_df, on='userID', how='inner')
    y = merged[col].values.astype(np.float32)
    X = merged.drop(columns=['userID', col]).values.astype(np.float32)
    return X, y


# ── Cohen's d ────────────────────────────────────────────────────────────────
def cohens_d_vector(X, y):
    pos, neg = X[y == 1], X[y == 0]
    diff = pos.mean(0) - neg.mean(0)
    n1, n2 = len(pos), len(neg)
    pooled = np.sqrt(((n1-1)*pos.var(0) + (n2-1)*neg.var(0)) / (n1+n2-2) + 1e-12)
    return diff / pooled


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


# ── ICL inference — numpy-vectorized, bypasses O(N²) dataset class ───────────
def icl_eval_multiseed(model, X, y, device, n_seeds=N_SEEDS,
                       ctx_size=CTX_SIZE, batch_size=128):
    """
    Direct numpy context sampling: avoids the O(N) Python list comprehension
    per query that makes InContextDiseaseDatasetNoID O(N²) at large N.
    Matches the legacy context_pos_ratio = min(prevalence*3, 0.3) behaviour.
    """
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y_int = y.astype(np.int32)
    pos_idx = np.where(y_int == 1)[0]
    neg_idx = np.where(y_int == 0)[0]
    N = len(y)

    prevalence = len(pos_idx) / max(N, 1)
    target_ratio = min(prevalence * 3, 0.3)
    n_ctx_pos = int(ctx_size * target_ratio)
    n_ctx_neg = ctx_size - n_ctx_pos
    can_stratify = len(pos_idx) > n_ctx_pos and len(neg_idx) > n_ctx_neg

    D = X.shape[1]
    # Pre-allocate once — avoids thousands of large malloc/free per disease
    ctx_prot_buf = np.empty((batch_size, ctx_size, D), dtype=np.float32)
    ctx_lbl_buf  = np.empty((batch_size, ctx_size),    dtype=np.float32)

    all_aurocs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        probs_all, tgts_all = [], []

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            bs = end - start
            # Views into the pre-allocated buffers (contiguous, zero-copy)
            ctx_prot = ctx_prot_buf[:bs]
            ctx_lbl  = ctx_lbl_buf[:bs]

            for i, qi in enumerate(range(start, end)):
                if can_stratify:
                    # Exclude query from its own class pool (numpy boolean mask — fast)
                    p_pool = pos_idx[pos_idx != qi] if y_int[qi] == 1 else pos_idx
                    n_pool = neg_idx[neg_idx != qi] if y_int[qi] == 0 else neg_idx
                    if len(p_pool) >= n_ctx_pos and len(n_pool) >= n_ctx_neg:
                        ci = np.concatenate([
                            rng.choice(p_pool, n_ctx_pos, replace=False),
                            rng.choice(n_pool, n_ctx_neg, replace=False),
                        ])
                    else:
                        avail = np.arange(N); avail = avail[avail != qi]
                        ci = rng.choice(avail, ctx_size,
                                        replace=(len(avail) < ctx_size))
                else:
                    avail = np.arange(N); avail = avail[avail != qi]
                    ci = rng.choice(avail, ctx_size,
                                    replace=(len(avail) < ctx_size))
                rng.shuffle(ci)
                ctx_prot[i] = X[ci]
                ctx_lbl[i]  = y[ci]

            batch = {
                'context_proteins': torch.from_numpy(ctx_prot).to(device),
                'context_labels':   torch.from_numpy(ctx_lbl).to(device),
                'query_proteins':   torch.from_numpy(X[start:end]).to(device),
                'target_label':     torch.from_numpy(y[start:end]).to(device),
            }
            with torch.no_grad():
                logits = model(batch)
            probs_all.extend(torch.sigmoid(logits).cpu().numpy())
            tgts_all.extend(y[start:end])

        try:
            all_aurocs.append(float(roc_auc_score(tgts_all, probs_all)))
        except Exception:
            pass

    if not all_aurocs:
        return float('nan'), 0.0, []
    return float(np.mean(all_aurocs)), float(np.std(all_aurocs)), all_aurocs


# ── LR + XGB baseline ────────────────────────────────────────────────────────
def run_baselines(X, y, n_splits=5):
    y = y.astype(int)
    if y.sum() < n_splits or (y == 0).sum() < n_splits:
        return float('nan'), float('nan'), float('nan')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    lr = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('sc', StandardScaler()),
                   ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    lr_scores = cross_val_score(lr, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.5,
                         use_label_encoder=False, eval_metric='logloss',
                         random_state=42, n_jobs=1, verbosity=0,
                         scale_pos_weight=float((y==0).sum()) / max(y.sum(), 1))
    xgb_scores = cross_val_score(xgb, X, y, cv=skf, scoring='roc_auc', n_jobs=1)
    return float(lr_scores.mean()), float(xgb_scores.mean()), float(xgb_scores.mean() - lr_scores.mean())


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def run_block(block, train_codes, exist_test, excl_prefixes, ckpt_path):
    print(f"\n{'='*70}")
    print(f"BLOCK {block}  checkpoint: {ckpt_path}")
    print(f"{'='*70}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = InContextTransformerGPT(protein_dim=2941, hidden_dim=768,
                                     n_layers=12, n_heads=12, dropout=0.2, temperature=1.0)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print("Model loaded.")

    # Load existing pkl for training diseases + existing test diseases
    with open(os.path.join(PROJECT, f'processed_data_{block.lower()}/train_data.pkl'), 'rb') as f:
        train_pkl = pickle.load(f)
    with open(os.path.join(PROJECT, f'processed_data_{block.lower()}/test_data.pkl'), 'rb') as f:
        test_pkl = pickle.load(f)
    all_pkl = {**train_pkl, **test_pkl}

    # Compute training disease Cohen's d vectors
    print("Computing training disease Cohen's d vectors...")
    train_d_vecs = {}
    for code in train_codes:
        if code not in all_pkl:
            print(f"  WARNING: {code} not in pkl")
            continue
        X, y = all_pkl[code]
        train_d_vecs[code] = cohens_d_vector(X, y.astype(int))
        print(f"  {code}: N={len(y)}, N+={int(y.sum())}")

    # ── Step 1: noise floor on existing test diseases ─────────────────────
    print(f"\n--- Step 1: Noise floor ({N_SEEDS} seeds × existing {block} test diseases) ---")
    noise_file = f'{OUT}/noise_{block}.json'
    if os.path.exists(noise_file):
        print(f"  Resuming: loading saved Step 1 from {noise_file}")
        with open(noise_file) as f:
            noise_results = json.load(f)
        for code, res in noise_results.items():
            print(f"  {code}: mean={res['mean']:.4f} ± CI95={res['ci95']:.4f}  (std={res['std']:.4f})")
    else:
        noise_results = {}
        for code in exist_test:
            if code not in all_pkl:
                continue
            X, y = all_pkl[code]
            mean_a, std_a, all_a = icl_eval_multiseed(model, X, y, device)
            ci95 = 1.96 * std_a / np.sqrt(len(all_a))
            noise_results[code] = {'mean': mean_a, 'std': std_a, 'ci95': ci95, 'seeds': all_a}
            print(f"  {code}: mean={mean_a:.4f} ± CI95={ci95:.4f}  (std={std_a:.4f})")
        with open(noise_file, 'w') as f:
            json.dump(noise_results, f, indent=2)

    # ── Step 2: build candidate pool from binary_csv ──────────────────────
    print(f"\n--- Step 2: Building candidate pool from binary_csv ---")
    protein_df = load_protein_data()
    binary_df  = load_binary_csv()

    # Binary CSV columns use format: Union#CODE#Description
    # Build {ICD_code: full_col_name} for this block, skipping block-level ranges
    code_map = {}
    for col in binary_df.columns:
        if not col.startswith('Union#'):
            continue
        parts = col.split('#')
        if len(parts) < 2:
            continue
        raw_code = parts[1]
        if 'Block' in raw_code or '-' in raw_code or raw_code.startswith('Chapter'):
            continue
        if not raw_code.startswith(block):
            continue
        code_map[raw_code] = col

    print(f"Found {len(code_map)} {block}* individual disease codes in binary_csv")

    # Exclusion rules
    excluded_train = set(train_codes)
    def is_excluded(code):
        if code in excluded_train:
            return True
        for pfx in excl_prefixes:
            if code.startswith(pfx) and code != pfx:
                return True
        return False

    # Build pool data: extract (X, y), filter N+
    pool_data  = {}   # code → (X, y)
    pool_stats = {}   # code → n, n_pos

    for code, col_name in code_map.items():
        if is_excluded(code):
            continue
        label_df = binary_df[['userID', col_name]].dropna()
        merged = protein_df.merge(label_df, on='userID', how='inner')
        y = merged[col_name].values.astype(np.float32)
        n_pos = int(y.sum())
        if n_pos < N_POS_MIN:
            continue
        X = merged.drop(columns=['userID', col_name]).values.astype(np.float32)
        # Fill NaN/Inf with 0 (data appears normalized; 0 ≈ population mean)
        if not np.isfinite(X).all():
            X[~np.isfinite(X)] = 0.0
        pool_data[code]  = (X, y)
        pool_stats[code] = {'n': len(y), 'n_pos': n_pos, 'prevalence': float(y.mean())}
        print(f"  {code}: N={len(y)}, N+={n_pos}")

    # Also include existing test diseases (already have them from pkl)
    for code in exist_test:
        if code in all_pkl and code not in pool_data:
            X, y = all_pkl[code]
            pool_data[code]  = (X, y)
            pool_stats[code] = {'n': len(y), 'n_pos': int(y.sum()), 'prevalence': float(y.mean())}

    print(f"\nTotal pool: {len(pool_data)} diseases "
          f"({len(exist_test)} existing + {len(pool_data)-len(exist_test)} new)")

    # ── Step 3: cosine + LR + XGB for each pool disease ──────────────────
    print(f"\n--- Step 3: Predictors (cosine + LR + XGB) ---")
    predictors = {}
    for code, (X, y) in pool_data.items():
        d_vec = cohens_d_vector(X, y.astype(int))
        sims  = [cosine(d_vec, train_d_vecs[tr]) for tr in train_codes if tr in train_d_vecs]
        cos_max  = float(np.max(sims))
        cos_mean = float(np.mean(sims))
        lr_a, xgb_a, gap = run_baselines(X, y)
        predictors[code] = {
            'cosine_max': cos_max, 'cosine_mean': cos_mean,
            'lr_auroc': lr_a, 'xgb_auroc': xgb_a, 'nonlin_gap': gap,
            'n_pos': pool_stats[code]['n_pos'],
            'existing': code in exist_test,
        }
        print(f"  {code}: cosine_max={cos_max:.3f} cosine_mean={cos_mean:.3f} "
              f"LR={lr_a:.3f} XGB={xgb_a:.3f}")

    with open(f'{OUT}/predictors_{block}.json', 'w') as f:
        json.dump(predictors, f, indent=2)

    # ── Out-of-sample: freeze rule from existing 5 diseases ──────────────
    exist_cos  = [predictors[c]['cosine_mean'] for c in exist_test if c in predictors]
    exist_icl  = [noise_results[c]['mean']     for c in exist_test if c in noise_results]
    if len(exist_cos) >= 3:
        slope, intercept, _, _, _ = stats.linregress(exist_cos, exist_icl)
        print(f"\nFrozen linear rule (from existing {len(exist_cos)} diseases):")
        print(f"  ICL_abs ≈ {intercept:.4f} + {slope:.4f} × cosine_mean")
        # Write predictions for all new diseases BEFORE running their ICL eval
        predictions = {}
        for code, pred in predictors.items():
            if code not in exist_test:
                predictions[code] = intercept + slope * pred['cosine_mean']
        with open(f'{OUT}/predictions_{block}.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved for {len(predictions)} new diseases.")
    else:
        slope, intercept = None, None
        predictions = {}

    # ── Step 4: ICL inference on all pool diseases (multi-seed) ──────────
    print(f"\n--- Step 4: ICL inference ({N_SEEDS} seeds × {len(pool_data)} diseases) ---")
    icl_file = f'{OUT}/icl_results_{block}.json'
    icl_results = {}

    # Resume from partial results if available
    if os.path.exists(icl_file):
        with open(icl_file) as f:
            icl_results = json.load(f)
        n_done = sum(1 for c in icl_results if c not in exist_test)
        print(f"  Resuming: {n_done} new diseases already done")
        for code, res in icl_results.items():
            if code not in exist_test:
                print(f"    {code}: mean={res['mean']:.4f} ± CI95={res['ci95']:.4f}")

    # Existing diseases — reuse noise_results
    for code in exist_test:
        if code in noise_results and code not in icl_results:
            icl_results[code] = {
                'mean': noise_results[code]['mean'],
                'std':  noise_results[code]['std'],
                'ci95': noise_results[code]['ci95'],
                'seeds': noise_results[code]['seeds'],
            }

    # New diseases
    for code, (X, y) in pool_data.items():
        if code in icl_results:
            continue
        print(f"  {code}...", flush=True)
        mean_a, std_a, all_a = icl_eval_multiseed(model, X, y, device)
        ci95 = 1.96 * std_a / np.sqrt(max(len(all_a), 1))
        icl_results[code] = {'mean': mean_a, 'std': std_a, 'ci95': ci95, 'seeds': all_a}
        print(f"    ICL_abs={mean_a:.4f} ± CI95={ci95:.4f}")
        # Save after each disease so a crash doesn't lose progress
        with open(icl_file, 'w') as f:
            json.dump(icl_results, f, indent=2)

    with open(icl_file, 'w') as f:
        json.dump(icl_results, f, indent=2)

    # ── Step 5: Analysis ──────────────────────────────────────────────────
    print(f"\n--- Step 5: Analysis ---")
    all_codes = [c for c in pool_data if c in icl_results and c in predictors]

    cos_vals = [predictors[c]['cosine_mean'] for c in all_codes]
    icl_vals = [icl_results[c]['mean']       for c in all_codes]
    ci95_vals = [icl_results[c]['ci95']      for c in all_codes]
    baseline_vals = [predictors[c]['xgb_auroc'] for c in all_codes]

    rho, p = stats.spearmanr(cos_vals, icl_vals)
    print(f"\nBlock {block} — cosine_mean vs ICL_abs: Spearman ρ={rho:.3f}, p={p:.3f} (N={len(all_codes)})")

    # Out-of-sample
    if predictions:
        new_codes = [c for c in all_codes if c not in exist_test and c in predictions]
        pred_vals   = [predictions[c]          for c in new_codes]
        actual_vals = [icl_results[c]['mean']  for c in new_codes]
        if len(new_codes) >= 3:
            rho_oos, p_oos = stats.spearmanr(pred_vals, actual_vals)
            print(f"Out-of-sample (N={len(new_codes)}): Spearman ρ={rho_oos:.3f}, p={p_oos:.3f}")
        else:
            rho_oos, p_oos = float('nan'), float('nan')
    else:
        new_codes, pred_vals, actual_vals = [], [], []
        rho_oos, p_oos = float('nan'), float('nan')

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    is_exist = [c in exist_test for c in all_codes]
    colors = ['#e74c3c' if e else '#3498db' for e in is_exist]
    ax.errorbar(cos_vals, icl_vals, yerr=ci95_vals, fmt='none', color='gray', alpha=0.4, zorder=1)
    ax.scatter(cos_vals, icl_vals, c=colors, s=60, zorder=2)
    for c, cx, cy in zip(all_codes, cos_vals, icl_vals):
        ax.annotate(c, (cx, cy), textcoords='offset points', xytext=(3, 2), fontsize=6)
    if slope is not None:
        xs = np.linspace(min(cos_vals), max(cos_vals), 100)
        ax.plot(xs, intercept + slope * xs, 'k--', lw=1, alpha=0.5, label='frozen rule')
    ax.set_xlabel('cosine_mean (to training diseases)')
    ax.set_ylabel('ICL_abs AUROC (mean ± 95%CI)')
    ax.set_title(f'Block {block}: cosine → ICL_abs\nSpearman ρ={rho:.3f}, p={p:.3f}, N={len(all_codes)}')
    ax.legend(handles=[
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#e74c3c', label='existing'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#3498db', label='new'),
    ])
    ax.grid(alpha=0.3)

    ax = axes[1]
    if new_codes:
        ax.scatter(pred_vals, actual_vals, c='#3498db', s=60, zorder=2)
        for c, px, ay in zip(new_codes, pred_vals, actual_vals):
            ax.annotate(c, (px, ay), textcoords='offset points', xytext=(3, 2), fontsize=6)
        mn = min(min(pred_vals), min(actual_vals)) - 0.02
        mx = max(max(pred_vals), max(actual_vals)) + 0.02
        ax.plot([mn, mx], [mn, mx], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Predicted ICL_abs (frozen linear rule)')
        ax.set_ylabel('Actual ICL_abs')
        ax.set_title(f'Block {block}: Out-of-sample\nSpearman ρ={rho_oos:.3f}, p={p_oos:.3f}, N={len(new_codes)}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/expanded_{block}.png', dpi=150)
    plt.close()

    # Save summary
    summary = {
        'block': block, 'n_total': len(all_codes),
        'n_existing': sum(is_exist), 'n_new': sum(1 for e in is_exist if not e),
        'spearman_rho': rho, 'spearman_p': p,
        'oos_rho': rho_oos, 'oos_p': p_oos,
        'slope': slope, 'intercept': intercept,
        'per_disease': {
            c: {'cosine_mean': predictors[c]['cosine_mean'],
                'icl_abs': icl_results[c]['mean'],
                'ci95': icl_results[c]['ci95'],
                'lr_auroc': predictors[c]['lr_auroc'],
                'xgb_auroc': predictors[c]['xgb_auroc'],
                'n_pos': predictors[c]['n_pos'],
                'existing': predictors[c]['existing']}
            for c in all_codes
        }
    }
    with open(f'{OUT}/summary_{block}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {OUT}/")
    return summary


if __name__ == '__main__':
    results = {}
    for block, train, exist, excl, ckpt in [
        ('C', TRAIN_C, EXIST_C, EXCL_PREFIX_C,
         f'{PROJECT}/checkpoints_c_model_v5/best_model.pt'),
        ('G', TRAIN_G, EXIST_G, EXCL_PREFIX_G,
         f'{PROJECT}/checkpoints_g_model_v5/best_model.pt'),
    ]:
        results[block] = run_block(block, train, exist, excl, ckpt)

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for block, r in results.items():
        print(f"Block {block}: N={r['n_total']}  "
              f"Spearman ρ={r['spearman_rho']:.3f} p={r['spearman_p']:.3f}  "
              f"OOS ρ={r['oos_rho']:.3f} p={r['oos_p']:.3f}")
