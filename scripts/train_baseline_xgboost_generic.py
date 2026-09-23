#!/usr/bin/env python -u
"""Generic XGBoost baseline — 15 random-seed re-splits per disease.

Each seed picks a fresh 80/20 stratified split and re-fits XGBoost. Per-disease
metrics reported as mean over the surviving seeds (nan seeds dropped), plus a
per-seed list preserved in the output JSON for downstream analysis.
"""
import sys, os, argparse, json, pickle
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.insert(0, str(Path(__file__).parent))
from metrics import auroc, auprc, brier, ece

N_SEEDS = 15
METRICS = ["auroc", "auprc", "brier", "ece"]


def fit_and_score(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    scale_pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.01,
        scale_pos_weight=scale_pos_weight, random_state=seed,
        eval_metric='logloss')
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    preds = (probs > 0.5).astype(int)
    return {
        'auroc': auroc(y_te, probs), 'auprc': auprc(y_te, probs),
        'brier': brier(y_te, probs), 'ece':   ece(y_te, probs),
        'accuracy':  float(accuracy_score(y_te, preds)),
        'precision': float(precision_score(y_te, preds, zero_division=0)),
        'recall':    float(recall_score(y_te, preds, zero_division=0)),
        'f1':        float(f1_score(y_te, preds, zero_division=0)),
        'n_test':    int(len(y_te)),
        'n_test_pos':int(y_te.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--diseases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block', default='')
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    project_root = os.environ.get('ICL_PROJECT_ROOT',
                                  str(Path(__file__).resolve().parent.parent))
    print(f"\n{'='*60}\nXGBoost Baseline: {args.block} block "
          f"({args.n_seeds} seeds)\n{'='*60}\n", flush=True)

    all_data = {}
    for split in ['train_data.pkl', 'test_data.pkl']:
        p = os.path.join(project_root, args.data_dir, split)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                all_data.update(pickle.load(f))

    results = {}
    for disease in args.diseases:
        print(f"\n{disease}", flush=True)
        if disease not in all_data:
            results[disease] = {'error': 'not found'}
            continue
        X, y = all_data[disease]
        print(f"Samples: {len(y)}, Positives: {int(y.sum())}", flush=True)
        if y.sum() < 5:
            results[disease] = {'error': 'insufficient positives'}
            continue

        per_seed = []
        for s in range(args.n_seeds):
            try:
                per_seed.append(fit_and_score(X, y, s))
            except Exception as e:
                print(f"  seed {s} failed: {e}", flush=True)

        def _mean(k):
            vals = [d[k] for d in per_seed
                    if d.get(k) is not None
                    and not (isinstance(d[k], float) and np.isnan(d[k]))]
            return float(np.mean(vals)) if vals else float('nan')

        def _std(k):
            vals = [d[k] for d in per_seed
                    if d.get(k) is not None
                    and not (isinstance(d[k], float) and np.isnan(d[k]))]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        results[disease] = {
            'n_total': len(y),
            'n_seeds': args.n_seeds,
            'n_seeds_valid': len(per_seed),
            # per-metric mean + std over seeds
            **{k: _mean(k) for k in METRICS},
            **{f'{k}_std': _std(k) for k in METRICS},
            'accuracy':  _mean('accuracy'),
            'precision': _mean('precision'),
            'recall':    _mean('recall'),
            'f1':        _mean('f1'),
            'per_seed':  per_seed,
        }
        r = results[disease]
        print(f"  AUROC: {r['auroc']:.4f}±{r['auroc_std']:.4f}  "
              f"AUPRC: {r['auprc']:.4f}±{r['auprc_std']:.4f}  "
              f"Brier: {r['brier']:.4f}±{r['brier_std']:.4f}  "
              f"ECE: {r['ece']:.4f}±{r['ece_std']:.4f}", flush=True)

    # Block summary
    def _block_mean(k):
        vals = [r[k] for r in results.values()
                if isinstance(r, dict) and r.get(k) is not None
                and not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(vals)) if vals else None

    results['summary'] = {
        'block': args.block,
        'n_seeds': args.n_seeds,
        'mean_auroc': _block_mean('auroc'), 'mean_auprc': _block_mean('auprc'),
        'mean_brier': _block_mean('brier'), 'mean_ece':   _block_mean('ece'),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
