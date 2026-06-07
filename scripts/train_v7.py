"""
V7 training: proxy validation disease for rigorous checkpoint selection.

Key changes vs v6:
- proxy_val_disease: one training disease held out from training,
  used for checkpoint selection (cross-disease AUROC signal, no test leakage)
- All remaining training data used for training (no 90/10 split)
- Early stopping based on proxy val AUROC, not in-domain val AUROC
- Cross-block eval on test diseases every N epochs (monitoring only, not selection)
- 13 training diseases + 4 new G diseases (G10/G41/G71/G80)
- epochs=40, patience=7
"""

import os, sys, argparse, yaml, pickle, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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


def crossblock_eval(model, data_dict, context_size, device, diseases,
                    max_eval_per_disease=5000):
    """Evaluate model on given diseases. Returns per-disease AUROC dict."""
    model.eval()
    results = {}
    for disease in diseases:
        if disease not in data_dict:
            continue
        X, y = data_dict[disease]
        if y.sum() < 5:
            continue
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

    proxy_val_disease = cfg['data']['proxy_val_disease']
    train_diseases = cfg['data']['train_diseases']
    test_diseases = cfg['data']['test_diseases']

    print(f"\n{'='*60}")
    print(f"V7 Training — {cfg['data']['processed_dir']}")
    print(f"Train diseases ({len(train_diseases)}): {train_diseases}")
    print(f"Proxy val disease: {proxy_val_disease}")
    print(f"Test diseases: {test_diseases}")
    print(f"Device: {device}")
    print(f"{'='*60}\n", flush=True)

    # Load data
    with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
        all_train = pickle.load(f)
    with open(os.path.join(processed_dir, 'test_data.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    # Separate proxy val from training
    if proxy_val_disease not in all_train:
        raise ValueError(f"proxy_val_disease '{proxy_val_disease}' not found in train pkl")
    proxy_val_data = {proxy_val_disease: all_train[proxy_val_disease]}

    train_data = {k: v for k, v in all_train.items() if k in train_diseases}

    # Clean NaN/Inf
    for code in list(train_data.keys()):
        X, y = train_data[code]
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        if n_nan > 0 or n_inf > 0:
            print(f"  {code}: {n_nan} NaN, {n_inf} Inf → replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            train_data[code] = (X, y)

    # Clean proxy val too
    Xp, yp = proxy_val_data[proxy_val_disease]
    n_nan = int(np.isnan(Xp).sum())
    if n_nan > 0 or int(np.isinf(Xp).sum()) > 0:
        print(f"  {proxy_val_disease} (proxy val): cleaning NaN/Inf")
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
        proxy_val_data[proxy_val_disease] = (Xp, yp)

    print("Training data:")
    for code, (X, y) in train_data.items():
        print(f"  {code}: {len(y)} samples, {int(y.sum())} pos ({y.mean():.4f})")
    Xp, yp = proxy_val_data[proxy_val_disease]
    print(f"Proxy val — {proxy_val_disease}: {len(yp)} samples, {int(yp.sum())} pos ({yp.mean():.4f})")

    missing = [d for d in train_diseases if d not in train_data]
    if missing:
        print(f"\nWARNING: missing training diseases: {missing}")

    # Class weights (training diseases only)
    class_weights = compute_class_weights(train_data)
    print(f"\nClass weights: { {k: f'{v:.3f}' for k, v in class_weights.items()} }")

    # Dataset — use ALL training data (no val split; proxy val disease is our selection signal)
    ctx_size = cfg['context']['context_size']
    sampling = cfg.get('sampling', {})
    context_pos_ratio = sampling.get('context_pos_ratio', 0.5)

    dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data,
        context_size=ctx_size,
        is_training=True,
        context_pos_ratio=context_pos_ratio,
        query_pos_ratio=None,
    )

    batch_size = cfg['training']['batch_size']
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn_no_id,
                              num_workers=cfg.get('num_workers', 4))

    print(f"\nDataset — {len(dataset)} training samples")
    print(f"Batches/epoch: {len(train_loader)}\n", flush=True)

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

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg['training']['learning_rate'],
                           weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['num_epochs'], eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    warmup_epochs = cfg['training'].get('warmup_epochs', 3)
    save_dir = os.path.join(project_root, cfg['training']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)

    patience = cfg['training']['early_stopping_patience']
    cb_interval = cfg['training'].get('crossblock_eval_interval', 3)

    best_proxy_auroc = 0.0
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
        elapsed = time.time() - t0

        # Proxy val eval — every epoch, used for checkpoint selection
        print(f"\n  [Proxy val eval @ epoch {epoch} — {proxy_val_disease}]", flush=True)
        proxy_results = crossblock_eval(model, proxy_val_data, ctx_size, device,
                                        [proxy_val_disease])
        proxy_auroc = proxy_results.get(proxy_val_disease, float('nan'))
        print(f"    {proxy_val_disease}: {proxy_auroc:.4f}", flush=True)

        # Cross-block eval on test diseases — monitoring only
        cb_aurocs = {}
        if epoch % cb_interval == 0 or epoch == 1:
            print(f"\n  [Cross-block eval @ epoch {epoch}]", flush=True)
            cb_aurocs = crossblock_eval(model, test_data, ctx_size, device, test_diseases)
            cb_mean = np.nanmean(list(cb_aurocs.values())) if cb_aurocs else float('nan')
            for d, a in cb_aurocs.items():
                print(f"    {d}: {a:.4f}", flush=True)
            print(f"    mean: {cb_mean:.4f}", flush=True)

        print(f"\n  Epoch {epoch:3d}  Train: {train_loss:.4f}  "
              f"ProxyVal({proxy_val_disease}): {proxy_auroc:.4f}  [{elapsed:.0f}s]", flush=True)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Save best based on PROXY VAL AUROC (not in-domain val)
        if proxy_auroc > best_proxy_auroc:
            best_proxy_auroc = proxy_auroc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'proxy_val_auroc': proxy_auroc,
                'crossblock_aurocs': cb_aurocs,
                'config': cfg,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"  Saved best model (proxy val AUROC: {proxy_auroc:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience {patience})", flush=True)
                break

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'proxy_val_auroc': proxy_auroc,
            'crossblock': cb_aurocs,
        })

    # Final cross-block eval with best model at ctx_size*2
    print(f"\n{'='*60}")
    print("Final cross-block evaluation with best model")
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    final_cb = crossblock_eval(model, test_data, ctx_size * 2, device, test_diseases)
    print(f"(ctx_size={ctx_size * 2} for final eval)")
    for d, a in final_cb.items():
        print(f"  {d}: {a:.4f}")
    print(f"  mean: {np.nanmean(list(final_cb.values())):.4f}", flush=True)

    out_path = os.path.join(project_root, 'results',
                            f"{cfg['data']['processed_dir']}_v7_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'final_crossblock': final_cb,
            'history': history,
            'best_proxy_val_auroc': best_proxy_auroc,
            'proxy_val_disease': proxy_val_disease,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
