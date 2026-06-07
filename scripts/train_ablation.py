"""
Causal ablation training for C44 removal experiment.
Identical to v5 architecture/hyperparams; only training disease list changes.
Supports per-disease sample cap for data-matching across conditions.
"""

import os, sys, argparse, yaml, pickle, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights
from training.trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def subsample_disease(X, y, n_max, seed=42):
    """Subsample (X, y) to n_max rows, stratified by label."""
    if len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    prev = y.mean()
    n_pos = min(len(pos_idx), round(n_max * prev))
    n_neg = n_max - n_pos
    n_neg = min(n_neg, len(neg_idx))
    n_pos = n_max - n_neg
    chosen = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False),
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    return X[chosen], y[chosen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   required=True)
    parser.add_argument('--save_dir', required=True,
                        help='checkpoint output directory')
    parser.add_argument('--max_samples_per_disease', type=str, default='',
                        help='comma-separated CODE:N pairs to cap specific diseases, '
                             'e.g. C44:29689')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['training']['save_dir'] = args.save_dir

    # Parse per-disease sample caps
    caps = {}
    if args.max_samples_per_disease:
        for tok in args.max_samples_per_disease.split(','):
            code, n = tok.strip().split(':')
            caps[code.strip()] = int(n)

    seed = config.get('seed', 42)
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    project_root = config['data']['project_root']
    processed_dir = os.path.join(project_root, config['data']['processed_dir'])

    with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
        all_train = pickle.load(f)

    train_diseases = config['data']['train_diseases']
    train_data = {}
    for code in train_diseases:
        X, y = all_train[code]
        if code in caps:
            X, y = subsample_disease(X, y, caps[code], seed=seed)
            print(f"  {code}: subsampled to {len(y)} (cap={caps[code]})")
        else:
            print(f"  {code}: {len(y)} samples (full)")
        train_data[code] = (X, y)

    total = sum(len(v[1]) for v in train_data.values())
    print(f"Total training samples: {total}")

    class_weights_dict = compute_class_weights(train_data)

    ctx_size = config['context']['context_size']
    dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data,
        context_size=ctx_size,
        is_training=True,
        context_pos_ratio=config['sampling'].get('context_pos_ratio'),
        query_pos_ratio=config['sampling'].get('query_pos_ratio'),
    )

    train_size = int(0.9 * len(dataset))
    val_size   = len(dataset) - train_size
    train_sub, val_sub = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_sub, batch_size=config['training']['batch_size'],
                              shuffle=True, collate_fn=collate_fn_no_id,
                              num_workers=config['num_workers'])
    val_loader   = DataLoader(val_sub,   batch_size=config['training']['batch_size'],
                              shuffle=False, collate_fn=collate_fn_no_id,
                              num_workers=config['num_workers'])

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = InContextTransformerGPT(
        protein_dim=config['model']['protein_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        temperature=config['model']['temperature'],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    os.makedirs(args.save_dir, exist_ok=True)
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                      config=config, device=device, save_dir=args.save_dir)
    trainer.train(class_weights=class_weights_dict)
    print(f"Done. Checkpoints saved to {args.save_dir}/")


if __name__ == '__main__':
    main()
