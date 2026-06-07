"""
DNN baseline (5-fold CV) for expanded validation diseases.
Only runs on diseases where ICL already beats both LR and XGB.
Architecture matches train_baseline_dnn.py: DNN(2941→1024→512→256→1, BN+ReLU+Dropout).
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT  = '/work/jl1401/icl_protein_disease'
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT      = os.path.join(PROJECT, 'analysis_results/expanded_validation')

# ── Target diseases (ICL beats both LR and XGB in expand_validation) ─────────
G_TARGET = ['G03','G24','G44','G479','G50','G51','G510','G530','G54','G55',
            'G551','G552','G57','G58','G589','G629','G909','G93','G933','G95','G98']
C_TARGET = ['C16','C180','C187','C189','C19','C20','C22','C25','C259','C435',
            'C436','C67','C679','C77','C770','C771','C772','C779','C78','C780',
            'C782','C785','C786','C787','C79','C793','C795','C798','C799','C80','C97']

N_SPLITS = 5
EPOCHS   = 60
LR       = 1e-3
BATCH    = 256
PATIENCE = 10


class DNN(nn.Module):
    def __init__(self, input_dim=2941, hidden_dims=(1024, 512, 256), dropout=0.3):
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


def run_dnn_cv(X, y, device):
    y = y.astype(int)
    if y.sum() < N_SPLITS or (y == 0).sum() < N_SPLITS:
        return float('nan'), []

    # NaN already handled in build_Xy (set to 0, same as expand_validation.py)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx].astype(np.float32), X[val_idx].astype(np.float32)
        y_tr, y_val = y[tr_idx].astype(np.float32), y[val_idx].astype(np.float32)

        # Per-fold feature standardisation (fit on train, apply to val)
        mean = X_tr.mean(0, keepdims=True)
        std  = X_tr.std(0, keepdims=True) + 1e-8
        X_tr  = (X_tr  - mean) / std
        X_val = (X_val - mean) / std

        pos_weight = torch.tensor(
            [(y_tr == 0).sum() / max(y_tr.sum(), 1)], dtype=torch.float32
        ).to(device)

        model = DNN(X_tr.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=BATCH, shuffle=True, pin_memory=True
        )
        X_val_t = torch.from_numpy(X_val).to(device)

        best_auc, best_state, patience_cnt = 0.0, None, 0

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                logits = model(X_val_t).cpu().numpy()
            try:
                auc = float(roc_auc_score(y_val, logits))
            except Exception:
                auc = 0.0

            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t).cpu().numpy()
        try:
            auc = float(roc_auc_score(y_val, logits))
        except Exception:
            auc = float('nan')
        fold_aucs.append(auc)
        print(f"    fold {fold+1}: {auc:.4f}", flush=True)

    return float(np.nanmean(fold_aucs)), fold_aucs


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
                             names=pd.read_csv(
                                 os.path.join(DATA_DIR, 'xaa.gz'),
                                 compression='gzip', nrows=0).columns)
        dfs.append(df)
        print(f"  {fname}: {len(df)} rows", flush=True)
    protein_df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {protein_df.shape}", flush=True)
    return protein_df


def load_binary_csv():
    print("Loading binary disease labels...", flush=True)
    df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'), compression='gzip')
    print(f"Binary CSV: {df.shape}", flush=True)
    # Build code → full column name map (format: Union#CODE#Description)
    code_map = {}
    for col in df.columns:
        if not col.startswith('Union#'):
            continue
        parts = col.split('#')
        if len(parts) >= 2:
            code_map[parts[1]] = col
    return df, code_map


def build_Xy(protein_df, binary_df, code_map, code):
    col = code_map.get(code)
    if col is None:
        return None, None
    label_df = binary_df[['userID', col]].dropna()
    merged   = protein_df.merge(label_df, on='userID', how='inner')
    y = merged[col].values.astype(np.float32)
    X = merged.drop(columns=['userID', col]).values.astype(np.float32)
    # Same NaN/Inf handling as expand_validation.py
    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = 0.0
    return X, y


def run_block(block, diseases, protein_df, binary_df, code_map, device, out_path):
    # Resume if partially done
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming {block}: {len(results)} already done", flush=True)
    else:
        results = {}

    # Load ICL/LR/XGB from summary for comparison
    with open(os.path.join(OUT, f'summary_{block}.json')) as f:
        summary = json.load(f)
    per = summary['per_disease']

    for code in diseases:
        if code in results:
            print(f"  {code}: already done (dnn={results[code]['dnn_auroc']:.4f}), skipping", flush=True)
            continue

        print(f"\n--- {block} {code} ---", flush=True)
        X, y = build_Xy(protein_df, binary_df, code_map, code)
        if X is None:
            print(f"  NOT FOUND in binary_csv, skipping", flush=True)
            results[code] = {'error': 'not found'}
            continue
        n_pos = int(y.sum())
        print(f"  N={len(y)}, pos={n_pos}", flush=True)

        dnn_mean, fold_aucs = run_dnn_cv(X, y, device)
        icl = per[code]['icl_abs']
        lr  = per[code]['lr_auroc']
        xgb = per[code]['xgb_auroc']

        results[code] = {
            'dnn_auroc': dnn_mean,
            'dnn_folds': fold_aucs,
            'icl_auroc': icl,
            'lr_auroc':  lr,
            'xgb_auroc': xgb,
            'icl_beats_dnn': bool(icl > dnn_mean),
            'n_pos': n_pos,
        }
        beat = "✓ ICL wins" if icl > dnn_mean else "✗ DNN wins"
        print(f"  DNN={dnn_mean:.4f}  ICL={icl:.4f}  {beat}", flush=True)

        # Save after every disease (safe resume)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def print_summary(block, results):
    valid = {k: v for k, v in results.items() if 'dnn_auroc' in v and not isinstance(v.get('dnn_auroc'), str)}
    beats = sum(1 for v in valid.values() if v['icl_beats_dnn'])
    total = len(valid)

    print(f"\n{'='*60}", flush=True)
    print(f"BLOCK {block}: ICL beats DNN in {beats}/{total} diseases", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Code':<8} {'ICL':>6}  {'LR':>6}  {'XGB':>6}  {'DNN':>6}  {'ICL>DNN':>8}", flush=True)

    rows = sorted(valid.items(), key=lambda x: x[1]['icl_auroc'] - x[1]['dnn_auroc'], reverse=True)
    for code, v in rows:
        flag = '✓' if v['icl_beats_dnn'] else '✗'
        print(f"{code:<8} {v['icl_auroc']:>6.4f}  {v['lr_auroc']:>6.4f}  {v['xgb_auroc']:>6.4f}  "
              f"{v['dnn_auroc']:>6.4f}    {flag}", flush=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"G diseases: {len(G_TARGET)}, C diseases: {len(C_TARGET)}", flush=True)

    protein_df = load_protein_data()
    binary_df, code_map  = load_binary_csv()

    g_out = os.path.join(OUT, 'dnn_baseline_G.json')
    c_out = os.path.join(OUT, 'dnn_baseline_C.json')
    # Remove stale empty output files from the failed run
    for p in [g_out, c_out]:
        if os.path.exists(p):
            import json as _j
            data = _j.load(open(p))
            if not any('dnn_auroc' in v for v in data.values()):
                os.remove(p)
                print(f"Removed stale {p}", flush=True)

    print("\n" + "="*60, flush=True)
    print("Block G", flush=True)
    print("="*60, flush=True)
    g_results = run_block('G', G_TARGET, protein_df, binary_df, code_map, device, g_out)
    print_summary('G', g_results)

    print("\n" + "="*60, flush=True)
    print("Block C", flush=True)
    print("="*60, flush=True)
    c_results = run_block('C', C_TARGET, protein_df, binary_df, code_map, device, c_out)
    print_summary('C', c_results)

    print(f"\nSaved: {g_out}", flush=True)
    print(f"Saved: {c_out}", flush=True)
    print("\n=== DONE ===", flush=True)


if __name__ == '__main__':
    main()
