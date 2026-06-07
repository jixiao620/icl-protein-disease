#!/usr/bin/env python -u
"""
Train ICL model on a single test disease (80/20 stratified split) and evaluate.

Diagnostic experiment: if ICL single-disease AUROC >= baseline, the bottleneck is
cross-block generalization, not the ICL approach itself.

Usage:
  python scripts/train_eval_single_disease.py \
    --disease C34 \
    --data_dir processed_data_c \
    --output results/single_disease_c.json \
    --context_pos_ratio 0.5 \
    --query_pos_ratio 0.5
"""
import sys, os, argparse, json, pickle, time
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights


def load_disease_data(data_dir, disease, project_root='/work/jl1401/icl_protein_disease'):
    """Load disease data from train_data.pkl and test_data.pkl (whichever has it)."""
    base = os.path.join(project_root, data_dir)
    for fname in ['train_data.pkl', 'test_data.pkl']:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if disease in d:
            return d[disease]
    raise ValueError(f"Disease {disease} not found in {data_dir}")


def eval_icl(model, test_data_dict, context_size, device, label=''):
    """Evaluate ICL model on test data using in-context inference."""
    model.eval()
    dataset = InContextDiseaseDatasetNoID(
        data_dict=test_data_dict,
        context_size=context_size,
        is_training=False,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False,
                        collate_fn=collate_fn_no_id, num_workers=0)

    all_probs, all_targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 200 == 0:
                print(f"  [{label}] Eval batch {i}/{len(loader)}", flush=True)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(batch['target_label'].cpu().numpy())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    auroc = float(roc_auc_score(all_targets, all_probs))
    auprc = float(average_precision_score(all_targets, all_probs))
    preds = (all_probs > 0.5).astype(int)
    f1 = float(f1_score(all_targets, preds, zero_division=0))
    return auroc, auprc, f1, len(all_targets), int(all_targets.sum())


def train_single_disease(disease, data_dir, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}", flush=True)
    print(f"Single-disease ICL: {disease}", flush=True)
    print(f"  data_dir: {data_dir}  device: {device}", flush=True)
    print(f"  context_pos_ratio={args.context_pos_ratio}  query_pos_ratio={args.query_pos_ratio}", flush=True)
    print(f"{'='*60}", flush=True)

    X_all, y_all = load_disease_data(data_dir, disease)
    print(f"Loaded {len(y_all)} samples, {int(y_all.sum())} positives ({y_all.mean()*100:.2f}%)", flush=True)

    if y_all.sum() < 10:
        print(f"WARNING: too few positives, skipping {disease}", flush=True)
        return {'error': 'too few positives', 'n_samples': len(y_all), 'n_positives': int(y_all.sum())}

    # 80/20 stratified split — same seed as DNN baseline for exact same test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"Train: {len(y_train)} ({int(y_train.sum())} pos)  Test: {len(y_test)} ({int(y_test.sum())} pos)", flush=True)

    train_data_dict = {disease: (X_train, y_train)}
    test_data_dict  = {disease: (X_test,  y_test)}

    class_weights = compute_class_weights(train_data_dict)
    print(f"Class weight for {disease}: {class_weights[disease]:.4f}", flush=True)

    train_dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data_dict,
        context_size=args.context_size,
        is_training=True,
        context_pos_ratio=args.context_pos_ratio,
        query_pos_ratio=args.query_pos_ratio,
    )
    n_train = int(0.9 * len(train_dataset))
    n_val   = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True,
                              collate_fn=collate_fn_no_id, num_workers=4)
    val_loader   = DataLoader(val_subset,   batch_size=8, shuffle=False,
                              collate_fn=collate_fn_no_id, num_workers=4)

    print(f"Dataset — train_subset: {n_train}  val_subset: {n_val}", flush=True)

    model = InContextTransformerGPT(
        protein_dim=2941, hidden_dim=768, n_layers=12, n_heads=12,
        dropout=0.2, temperature=1.0
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auroc = 0.0
    best_state = None
    patience = 0
    patience_limit = 15

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(batch)
            targets = batch['target_label']
            loss = criterion(logits, targets)

            # Class weights on positives only
            weights = torch.ones_like(loss)
            pos_w = class_weights.get(disease, 1.0)
            pos_mask = (targets == 1)
            weights[pos_mask] = pos_w
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_losses, val_probs, val_targets_list = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(batch)
                targets = batch['target_label']
                loss = criterion(logits, targets).mean()
                val_losses.append(loss.item())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_loss = np.mean(val_losses)
        try:
            val_auroc = roc_auc_score(val_targets_list, val_probs)
        except Exception:
            val_auroc = 0.0

        print(f"  Epoch {epoch:3d}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val AUROC: {val_auroc:.4f}", flush=True)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  Early stopping at epoch {epoch} (patience={patience_limit})", flush=True)
                break

    # Restore best model and evaluate on test set
    print(f"\nBest val AUROC: {best_val_auroc:.4f}  Evaluating on test set...", flush=True)
    model.load_state_dict(best_state)

    auroc, auprc, f1, n_te, n_pos = eval_icl(model, test_data_dict, args.context_size, device, disease)
    print(f"  Test AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}  F1: {f1:.4f}", flush=True)

    return {
        'disease': disease,
        'n_train': len(y_train), 'n_train_pos': int(y_train.sum()),
        'n_test':  len(y_test),  'n_test_pos':  int(y_test.sum()),
        'prevalence': float(y_all.mean()),
        'best_val_auroc': best_val_auroc,
        'test_auroc': auroc,
        'test_auprc': auprc,
        'test_f1': f1,
        'context_pos_ratio': args.context_pos_ratio,
        'query_pos_ratio': args.query_pos_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diseases', nargs='+', required=True,
                        help='Disease codes to train+eval (e.g. C34 C18 C43 C53 C54)')
    parser.add_argument('--data_dir', required=True,
                        help='processed_data_c or processed_data_g')
    parser.add_argument('--output', required=True)
    parser.add_argument('--context_size', type=int, default=32)
    parser.add_argument('--context_pos_ratio', type=float, default=0.5)
    parser.add_argument('--query_pos_ratio',   type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    all_results = {}
    for disease in args.diseases:
        t0 = time.time()
        result = train_single_disease(disease, args.data_dir, args)
        result['elapsed_min'] = (time.time() - t0) / 60
        all_results[disease] = result

    valid = [r['test_auroc'] for r in all_results.values() if 'test_auroc' in r]
    all_results['summary'] = {
        'mean_test_auroc': float(np.mean(valid)) if valid else None,
        'context_pos_ratio': args.context_pos_ratio,
        'query_pos_ratio': args.query_pos_ratio,
        'diseases': args.diseases,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY — Single-Disease ICL vs DNN Baseline", flush=True)
    print(f"{'='*60}", flush=True)
    for disease, r in all_results.items():
        if disease == 'summary':
            continue
        if 'error' in r:
            print(f"  {disease}: ERROR — {r['error']}", flush=True)
        else:
            print(f"  {disease}: ICL={r['test_auroc']:.4f}  (val best={r['best_val_auroc']:.4f})", flush=True)
    if valid:
        print(f"  Mean ICL: {np.mean(valid):.4f}", flush=True)


if __name__ == '__main__':
    main()
