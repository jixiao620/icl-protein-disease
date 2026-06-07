#!/usr/bin/env python -u
"""
DNN Baseline: per-disease training, within-disease 80/20 split.
Usage:
  python scripts/train_baseline_dnn.py \
    --data_dir processed_data_c \
    --diseases C34 C18 C43 C53 C54 \
    --output results/baseline_dnn_c.json \
    --block C
"""
import sys, os, argparse, json, pickle
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

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

def train_one_disease(X, y, device, epochs=50, lr=1e-3, batch_size=256):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max(y_tr.sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    best_auroc, best_probs = 0.0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_te_t)).cpu().numpy()
        try:
            auroc = roc_auc_score(y_te_t.numpy(), probs)
            if auroc > best_auroc:
                best_auroc = auroc
                best_probs = probs
        except:
            pass

    if best_probs is None:
        best_probs = probs
    return best_probs, y_te

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--diseases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block', default='')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = '/work/jl1401/icl_protein_disease'

    print(f"\n{'='*60}\nDNN Baseline: {args.block} block\nDevice: {device}\n{'='*60}\n", flush=True)

    # Load all data
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

        probs, y_te = train_one_disease(X, y, device)
        preds = (probs > 0.5).astype(int)

        try:
            auroc = float(roc_auc_score(y_te, probs))
        except:
            auroc = None
        try:
            auprc = float(average_precision_score(y_te, probs))
        except:
            auprc = None

        results[disease] = {
            'n_total': len(y), 'n_test': len(y_te), 'n_test_pos': int(y_te.sum()),
            'auroc': auroc, 'auprc': auprc,
            'accuracy': float(accuracy_score(y_te, preds)),
            'precision': float(precision_score(y_te, preds, zero_division=0)),
            'recall': float(recall_score(y_te, preds, zero_division=0)),
            'f1': float(f1_score(y_te, preds, zero_division=0))
        }
        print(f"  AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}", flush=True)

    valid = [r['auroc'] for r in results.values() if r.get('auroc') is not None]
    results['summary'] = {'mean_auroc': float(np.mean(valid)) if valid else None, 'block': args.block}
    print(f"\nMean AUROC: {results['summary']['mean_auroc']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}", flush=True)

if __name__ == '__main__':
    main()
