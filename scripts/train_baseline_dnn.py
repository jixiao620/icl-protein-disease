#!/usr/bin/env python -u
"""
DNN Baseline — 15 random-seed re-splits per disease.

Each seed picks a fresh 80/20 stratified split and re-fits the same DNN
architecture (Adam + cosine schedule + BCE with pos_weight). Per-disease
metrics are the mean over surviving seeds; per-seed values are preserved
in the output JSON for downstream analysis.
Usage:
  python scripts/train_baseline_dnn.py \\
    --data_dir processed_data_c \\
    --diseases C34 C18 C43 C53 C54 \\
    --output results/baseline_dnn_c.json \\
    --block C  [--n_seeds 15]
"""
import sys, os, argparse, json, pickle
from pathlib import Path
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
sys.path.insert(0, str(Path(__file__).parent))
from metrics import auroc, auprc, brier, ece

N_SEEDS = 15
METRICS = ["auroc", "auprc", "brier", "ece"]


class DNN(nn.Module):
    def __init__(self, input_dim=2941, hidden_dims=[1024, 512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_seed(X, y, device, seed, epochs=50, lr=1e-3, batch_size=256):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # 3-way split: 60% train / 20% val (early-stop selection) / 20% test (reported once)
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.25, random_state=seed, stratify=y_trv)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max(y_tr.sum(), 1)],
                              dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_va_np = y_va
    y_te_np = y_te

    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=batch_size, shuffle=True)

    best_val_auroc, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_va_t)).cpu().numpy()
        try:
            va = roc_auc_score(y_va_np, val_probs)
            if va > best_val_auroc:
                best_val_auroc = va
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        except Exception:
            pass

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
    preds = (best_probs > 0.5).astype(int)
    return {
        'auroc': auroc(y_te_np, best_probs),
        'auprc': auprc(y_te_np, best_probs),
        'brier': brier(y_te_np, best_probs),
        'ece':   ece(y_te_np, best_probs),
        'accuracy':  float(accuracy_score(y_te_np, preds)),
        'precision': float(precision_score(y_te_np, preds, zero_division=0)),
        'recall':    float(recall_score(y_te_np, preds, zero_division=0)),
        'f1':        float(f1_score(y_te_np, preds, zero_division=0)),
        'n_test':    int(len(y_te_np)),
        'n_test_pos':int(y_te_np.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--diseases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block', default='')
    parser.add_argument('--n_seeds', type=int, default=N_SEEDS)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.environ.get('ICL_PROJECT_ROOT',
                                  str(Path(__file__).resolve().parent.parent))

    print(f"\n{'='*60}\nDNN Baseline: {args.block} block "
          f"({args.n_seeds} seeds)\nDevice: {device}\n{'='*60}\n", flush=True)

    all_data = {}
    for split in ['train_data.pkl', 'test_data.pkl']:
        p = os.path.join(project_root, args.data_dir, split)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                all_data.update(pickle.load(f))

    results = {}
    for disease in args.diseases:
        print(f"\n{'='*40}\n{disease}", flush=True)
        if disease not in all_data:
            print(f"WARNING: {disease} not found, skipping.", flush=True)
            results[disease] = {'error': 'not found'}
            continue
        X, y = all_data[disease]
        print(f"Samples: {len(y)}, Positives: {int(y.sum())}", flush=True)
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            print(f"WARNING: insufficient samples, skipping.", flush=True)
            results[disease] = {'error': 'insufficient samples'}
            continue

        per_seed = []
        for s in range(args.n_seeds):
            try:
                per_seed.append(train_one_seed(X, y, device, s))
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
    s = results['summary']
    print(f"\nBlock summary — mean over test diseases:\n"
          f"  AUROC: {s['mean_auroc']}  AUPRC: {s['mean_auprc']}  "
          f"Brier: {s['mean_brier']}  ECE: {s['mean_ece']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
