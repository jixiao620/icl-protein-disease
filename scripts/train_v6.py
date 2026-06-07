"""
V6 training: expanded training diseases, no query oversampling, periodic cross-block eval.

Key changes vs v5:
- 14 training diseases (was 4)
- No query_pos_ratio oversampling → ~7x faster epochs
- Cross-block eval on test diseases every N epochs during training
- batch_size=16, patience=7, epochs=25
"""

import os, sys, argparse, yaml, pickle, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random; random.seed(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def crossblock_eval(model, test_data, context_size, device, diseases,
                    max_eval_per_disease=5000):
    """Quick cross-block eval on test diseases. Returns per-disease AUROC dict."""
    model.eval()
    results = {}
    for disease in diseases:
        if disease not in test_data:
            continue
        X, y = test_data[disease]
        if y.sum() < 5:
            continue
        # Subsample to speed up eval while preserving class ratio
        if len(y) > max_eval_per_disease:
            rng = np.random.default_rng(seed=42)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(len(pos_idx), max_eval_per_disease // 2)
            n_neg = min(len(neg_idx), max_eval_per_disease - n_pos)
            chosen = np.concatenate([
                rng.choice(pos_idx, n_pos, replace=False),
                rng.choice(neg_idx, n_neg, replace=False),
            ])
            X, y = X[chosen], y[chosen]
        dataset = InContextDiseaseDatasetNoID(
            data_dict={disease: (X, y)},
            context_size=context_size,
            is_training=False,
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=False,
                            collate_fn=collate_fn_no_id, num_workers=2)
        probs, targets = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = model(batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(batch['target_label'].cpu().numpy())
        try:
            auroc = float(roc_auc_score(targets, probs))
        except Exception:
            auroc = float('nan')
        results[disease] = auroc
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = cfg['data']['project_root']
    processed_dir = os.path.join(project_root, cfg['data']['processed_dir'])

    print(f"\n{'='*60}")
    print(f"V6 Training — {cfg['data']['processed_dir']}")
    print(f"Training diseases: {cfg['data']['train_diseases']}")
    print(f"Test diseases: {cfg['data']['test_diseases']}")
    print(f"Device: {device}")
    print(f"{'='*60}\n", flush=True)

    # Load data
    with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
        all_train = pickle.load(f)
    with open(os.path.join(processed_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    train_data = {k: v for k, v in all_train.items()
                  if k in cfg['data']['train_diseases']}

    # Clean NaN/Inf in protein features (some patients have missing measurements)
    for code in list(train_data.keys()):
        X, y = train_data[code]
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        if n_nan > 0 or n_inf > 0:
            print(f"  {code}: {n_nan} NaN, {n_inf} Inf → replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            train_data[code] = (X, y)

    print("Training data:")
    for code, (X, y) in train_data.items():
        print(f"  {code}: {len(y)} samples, {int(y.sum())} pos ({y.mean():.4f})")

    missing = [d for d in cfg['data']['train_diseases'] if d not in train_data]
    if missing:
        print(f"\nWARNING: missing training diseases: {missing}")
        print("These will be skipped. Check preprocessing output.")

    # Class weights
    class_weights = compute_class_weights(train_data)
    print(f"\nClass weights: { {k: f'{v:.3f}' for k, v in class_weights.items()} }")

    # Dataset (no query oversampling)
    ctx_size = cfg['context']['context_size']
    sampling = cfg.get('sampling', {})
    context_pos_ratio = sampling.get('context_pos_ratio', 0.5)
    query_pos_ratio = sampling.get('query_pos_ratio', None)

    dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data,
        context_size=ctx_size,
        is_training=True,
        context_pos_ratio=context_pos_ratio,
        query_pos_ratio=query_pos_ratio,
    )

    train_n = int(0.9 * len(dataset))
    val_n = len(dataset) - train_n
    train_sub, val_sub = random_split(dataset, [train_n, val_n],
                                      generator=torch.Generator().manual_seed(cfg['seed']))

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_no_id,
                              num_workers=cfg.get('num_workers', 4))
    val_loader = DataLoader(val_sub, batch_size=batch_size*2, shuffle=False,
                            collate_fn=collate_fn_no_id,
                            num_workers=cfg.get('num_workers', 4))

    print(f"\nDataset — train: {train_n}  val: {val_n}")
    print(f"Batches/epoch — train: {len(train_loader)}  val: {len(val_loader)}\n", flush=True)

    # Model
    model = InContextTransformerGPT(
        protein_dim=cfg['model']['protein_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        n_layers=12, n_heads=12,
        dropout=cfg['model']['dropout'],
        temperature=cfg['model']['temperature'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['training']['learning_rate'],
                           weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['num_epochs'], eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Warmup
    warmup_epochs = cfg['training'].get('warmup_epochs', 3)

    save_dir = os.path.join(project_root, cfg['training']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)

    patience = cfg['training']['early_stopping_patience']
    cb_interval = cfg['training'].get('crossblock_eval_interval', 3)
    test_diseases = cfg['data']['test_diseases']

    best_val_auroc = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batch = 0.0, 0

        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits = model(batch)
            targets = batch['target_label']
            loss = criterion(logits, targets)

            # class weights on positives
            weights = torch.ones_like(loss)
            for code, pw in class_weights.items():
                mask = torch.tensor([c == code for c in batch['query_disease_codes']],
                                    device=device)
                pos_mask = mask & (targets == 1)
                weights[pos_mask] = pw
            loss = (loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1
            if i % cfg['training']['log_interval'] == 0:
                print(f"   Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)

        train_loss = total_loss / n_batch

        # Validation (on oversampled val subset → val AUROC)
        model.eval()
        val_probs, val_targets, val_losses = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = model(batch)
                targets = batch['target_label']
                loss = criterion(logits, targets)
                weights = torch.ones_like(loss)
                for code, pw in class_weights.items():
                    mask = torch.tensor([c == code for c in batch['query_disease_codes']],
                                        device=device)
                    pos_mask = mask & (targets == 1)
                    weights[pos_mask] = pw
                val_losses.append((loss * weights).mean().item())
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_loss = float(np.mean(val_losses))
        try:
            val_auroc = float(roc_auc_score(val_targets, val_probs))
        except Exception:
            val_auroc = 0.0

        elapsed = time.time() - t0

        # Cross-block eval on actual test diseases
        cb_aurocs = {}
        if epoch % cb_interval == 0 or epoch == 1:
            print(f"\n  [Cross-block eval @ epoch {epoch}]", flush=True)
            cb_aurocs = crossblock_eval(model, test_data, ctx_size, device, test_diseases)
            cb_mean = np.mean(list(cb_aurocs.values())) if cb_aurocs else float('nan')
            for d, a in cb_aurocs.items():
                print(f"    {d}: {a:.4f}", flush=True)
            print(f"    mean: {cb_mean:.4f}", flush=True)

        print(f"\n  Epoch {epoch:3d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"ValAUROC: {val_auroc:.4f}  [{elapsed:.0f}s]", flush=True)

        # LR scheduler (skip warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        # Save best
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'crossblock_aurocs': cb_aurocs,
                'config': cfg,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  Saved best model (val AUROC: {val_auroc:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience {patience})", flush=True)
                break

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_auroc': val_auroc, 'crossblock': cb_aurocs,
        })

    # Final cross-block eval with best model
    print(f"\n{'='*60}")
    print("Final cross-block evaluation with best model")
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    final_cb = crossblock_eval(model, test_data, ctx_size * 2, device, test_diseases)
    print(f"(ctx_size={ctx_size*2} for final eval)")
    for d, a in final_cb.items():
        print(f"  {d}: {a:.4f}")
    print(f"  mean: {np.mean(list(final_cb.values())):.4f}", flush=True)

    out_path = os.path.join(project_root, 'results',
                            f"{cfg['data']['processed_dir']}_v6_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'final_crossblock': final_cb, 'history': history,
                   'best_val_auroc': best_val_auroc}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
