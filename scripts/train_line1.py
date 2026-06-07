"""
Generic training script for Line 1 pathway validation and other ICL experiments.
Reads all config params including sampling ratios (unlike train_gpt_ctx64.py).
"""

import os, sys, argparse, yaml, pickle
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights
from training.trainer import Trainer


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Config: {args.config}", flush=True)

    project_root = config['data']['project_root']
    processed_dir = os.path.join(project_root, config['data']['processed_dir'])

    with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)

    train_diseases = config['data']['train_diseases']
    train_data = {k: v for k, v in all_data.items() if k in train_diseases}
    print(f"\nTraining diseases: {list(train_data.keys())}", flush=True)
    for code, (X, y) in train_data.items():
        print(f"  {code}: N={len(y)}, N+={int(y.sum())} ({y.mean():.4f})", flush=True)

    class_weights = compute_class_weights(train_data)

    sampling = config.get('sampling', {})
    ctx_pos_ratio = sampling.get('context_pos_ratio', None)
    qry_pos_ratio = sampling.get('query_pos_ratio', None)

    dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data,
        context_size=config['context']['context_size'],
        is_training=True,
        context_pos_ratio=ctx_pos_ratio,
        query_pos_ratio=qry_pos_ratio,
    )

    train_n = int(0.9 * len(dataset))
    val_n = len(dataset) - train_n
    train_sub, val_sub = torch.utils.data.random_split(
        dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    num_workers = config.get('num_workers', 4)
    train_loader = DataLoader(
        train_sub, batch_size=config['training']['batch_size'],
        shuffle=True, collate_fn=collate_fn_no_id, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_sub, batch_size=config['training']['batch_size'],
        shuffle=False, collate_fn=collate_fn_no_id, num_workers=num_workers
    )
    print(f"\nTrain: {len(train_sub)}, Val: {len(val_sub)}", flush=True)

    model = InContextTransformerGPT(
        protein_dim=config['model']['protein_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        temperature=config['model']['temperature'],
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    save_dir = os.path.join(project_root, config['training']['save_dir'])
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=save_dir,
    )
    trainer.train(class_weights=class_weights)
    print(f"\nCheckpoints saved to: {save_dir}", flush=True)


if __name__ == '__main__':
    main()
