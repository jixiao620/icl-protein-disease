"""
Train on I10+I12+I120 with GPT2 backbone + all improvements
"""

import os
import sys
import argparse
import yaml
import pickle
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights
from training.trainer import Trainer

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def main(args):
    print("\n" + "="*60)
    print("🚀 Full Protein Training with GPT2 + All Improvements")
    print("="*60)
    
    config = load_config(args.config)
    set_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    project_root = config['data'].get('project_root', '.')
    processed_dir = os.path.join(project_root, config['data']['processed_dir'])
    
    # Load data
    with open(os.path.join(processed_dir, 'train_data.pkl'), 'rb') as f:
        all_train_data = pickle.load(f)
    
    train_data = {k: v for k, v in all_train_data.items() if k in config['data']['train_diseases']}
    
    print(f"\n✅ Training data (I10+I12+I120):")
    for code, (X, y) in train_data.items():
        print(f"   {code}: {X.shape[0]} samples, {y.sum()}/{len(y)} positive ({y.mean():.3f})")
    
    # Compute class weights
    class_weights_dict = compute_class_weights(train_data)
    print(f"\n📊 Class weights (log-flattened):")
    for code, weight in class_weights_dict.items():
        print(f"   {code}: {weight:.4f}")
    
    # Create datasets
    sampling_cfg = config.get('sampling', {})
    context_pos_ratio = sampling_cfg.get('context_pos_ratio', None)
    query_pos_ratio   = sampling_cfg.get('query_pos_ratio', None)

    train_dataset = InContextDiseaseDatasetNoID(
        data_dict=train_data,
        context_size=config['context']['context_size'],
        is_training=True,
        context_pos_ratio=context_pos_ratio,
        query_pos_ratio=query_pos_ratio,
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_no_id,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_no_id,
        num_workers=config['num_workers']
    )
    
    print(f"\n   Train: {len(train_subset)} samples")
    print(f"   Val: {len(val_subset)} samples")
    
    # Create GPT model
    model = InContextTransformerGPT(
        protein_dim=config['model']['protein_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        temperature=config['model']['temperature']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ GPT2 Model:")
    print(f"   Parameters: {n_params:,}")
    print(f"   Improvements:")
    print(f"     ✓ Log-flattened class weights")
    print(f"     ✓ Oversampled positives (3x)")
    print(f"     ✓ Softmax attention")
    print(f"     ✓ Pretrained GPT2")
    print(f"     ✓ Gradient clipping")
    
    # Train
    save_dir = os.path.join(project_root, config['training']['save_dir'])
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=save_dir
    )
    
    trainer.train(class_weights=class_weights_dict)
    
    print(f"\n📁 Checkpoints: {save_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/full_proteins_gpt.yaml')
    args = parser.parse_args()
    
    main(args)
