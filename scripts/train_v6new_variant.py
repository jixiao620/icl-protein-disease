"""
Training script for ICL Transformer v6 architectural variants.

Variant is chosen via `config['model']['variant']`:
  * "v6" (default)             → ICLTransformerV6 (baseline)
  * "diseaseemb"               → ICLTransformerV6DiseaseEmb (Exp A)
  * "perprotattn"              → ICLTransformerV6PerProtAttn (Exp B)
  * "diseaseemb_perprotattn"   → ICLTransformerV6DiseaseEmbPerProtAttn (A+B)

Everything else (data loader, trainer, resume) is identical to train_v6new.py.
"""

import os, sys, argparse, yaml, pickle
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_icl_v6 import ICLTransformerV6
from models.transformer_icl_v6_diseaseemb import (
    ICLTransformerV6DiseaseEmb, build_prefix_vocab,
)
from models.transformer_icl_v6_perprotattn import ICLTransformerV6PerProtAttn
from models.transformer_icl_v6_diseaseemb_perprotattn import ICLTransformerV6DiseaseEmbPerProtAttn
from models.transformer_icl_v7_kitchensink import ICLTransformerV7KitchenSink
from models.transformer_icl_v8_protid import ICLTransformerV8ProtID
from models.transformer_icl_v9_featattn import ICLTransformerV9FeatAttn
from models.transformer_icl_v9cls_featattn import ICLTransformerV9CLSFeatAttn
from models.transformer_icl_v10_protid_diseaseemb import ICLTransformerV10ProtIDDiseaseEmb
from models.transformer_icl_v11_featattn_diseaseemb import ICLTransformerV11FeatAttnDiseaseEmb
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from data.dataset import compute_class_weights
from training.trainer import Trainer


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(model_cfg: dict, config: dict, processed_dir: str):
    variant = model_cfg.get('variant', 'v6')

    common = dict(
        protein_dim     = model_cfg['protein_dim'],
        hidden_dim      = model_cfg['hidden_dim'],
        n_layers        = model_cfg.get('n_layers', 12),
        n_heads         = model_cfg.get('n_heads', 12),
        dropout         = model_cfg['dropout'],
        dim_feedforward = model_cfg.get('dim_feedforward', 3072),
    )

    if variant == 'v6':
        return ICLTransformerV6(**common), None

    if variant == 'perprotattn':
        return ICLTransformerV6PerProtAttn(
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
        ), None

    if variant == 'v7_kitchensink':
        return ICLTransformerV7KitchenSink(
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
        ), None

    if variant == 'v8_protid':
        return ICLTransformerV8ProtID(
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
        ), None

    if variant == 'v9_featattn':
        return ICLTransformerV9FeatAttn(
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
            n_cls_feat     =model_cfg.get('n_cls_feat', 24),
            n_feat_layers  =model_cfg.get('n_feat_layers', 3),
            feat_heads     =model_cfg.get('feat_heads', 4),
        ), None

    if variant == 'v9cls_featattn':
        return ICLTransformerV9CLSFeatAttn(
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
            n_cls_feat     =model_cfg.get('n_cls_feat', 24),
            n_feat_layers  =model_cfg.get('n_feat_layers', 1),
            feat_heads     =model_cfg.get('feat_heads', 4),
            cls_init_std   =model_cfg.get('cls_init_std', 0.5),
        ), None

    if variant == 'diseaseemb':
        # Build vocab from union of train + test diseases.
        # Train codes come from config; test codes come from the selection
        # (path in config['data']['selection_path']) so test codes are known
        # at train time and share the prefix embedding space with train.
        train_codes = list(config['data']['train_diseases'])
        test_codes  = []
        sel_path = config['data'].get('selection_path')
        if sel_path and os.path.exists(sel_path):
            sel = yaml.safe_load(open(sel_path))
            block = config['data'].get('block_letter')
            if block and block in sel.get('blocks', {}):
                test_codes = list(sel['blocks'][block].get('test', []))
        all_codes = sorted(set(train_codes) | set(test_codes))
        prefix_vocab, code_to_prefix_ids = build_prefix_vocab(all_codes)
        print(f"  Prefix vocab size: {len(prefix_vocab)}", flush=True)
        print(f"  Codes covered:     {len(all_codes)} (train={len(train_codes)}, test={len(test_codes)})", flush=True)
        return ICLTransformerV6DiseaseEmb(
            prefix_vocab=prefix_vocab,
            code_to_prefix_ids=code_to_prefix_ids,
            **common,
        ), {'prefix_vocab': prefix_vocab, 'code_to_prefix_ids': code_to_prefix_ids}

    if variant == 'diseaseemb_perprotattn':
        # Combined A+B: builds prefix vocab (Exp A) AND per-protein attn (Exp B).
        train_codes = list(config['data']['train_diseases'])
        test_codes  = []
        sel_path = config['data'].get('selection_path')
        if sel_path and os.path.exists(sel_path):
            sel = yaml.safe_load(open(sel_path))
            block = config['data'].get('block_letter')
            if block and block in sel.get('blocks', {}):
                test_codes = list(sel['blocks'][block].get('test', []))
        all_codes = sorted(set(train_codes) | set(test_codes))
        prefix_vocab, code_to_prefix_ids = build_prefix_vocab(all_codes)
        print(f"  Prefix vocab size: {len(prefix_vocab)}", flush=True)
        print(f"  Codes covered:     {len(all_codes)} (train={len(train_codes)}, test={len(test_codes)})", flush=True)
        return ICLTransformerV6DiseaseEmbPerProtAttn(
            prefix_vocab=prefix_vocab,
            code_to_prefix_ids=code_to_prefix_ids,
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
        ), {'prefix_vocab': prefix_vocab, 'code_to_prefix_ids': code_to_prefix_ids}

    if variant == 'v10_protid_diseaseemb':
        train_codes = list(config['data']['train_diseases'])
        test_codes  = []
        sel_path = config['data'].get('selection_path')
        if sel_path and os.path.exists(sel_path):
            sel = yaml.safe_load(open(sel_path))
            block = config['data'].get('block_letter')
            if block and block in sel.get('blocks', {}):
                test_codes = list(sel['blocks'][block].get('test', []))
        all_codes = sorted(set(train_codes) | set(test_codes))
        prefix_vocab, code_to_prefix_ids = build_prefix_vocab(all_codes)
        print(f"  Prefix vocab size: {len(prefix_vocab)}", flush=True)
        print(f"  Codes covered:     {len(all_codes)} (train={len(train_codes)}, test={len(test_codes)})", flush=True)
        return ICLTransformerV10ProtIDDiseaseEmb(
            prefix_vocab=prefix_vocab,
            code_to_prefix_ids=code_to_prefix_ids,
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
        ), {'prefix_vocab': prefix_vocab, 'code_to_prefix_ids': code_to_prefix_ids}

    if variant == 'v11_featattn_diseaseemb':
        train_codes = list(config['data']['train_diseases'])
        test_codes  = []
        sel_path = config['data'].get('selection_path')
        if sel_path and os.path.exists(sel_path):
            sel = yaml.safe_load(open(sel_path))
            block = config['data'].get('block_letter')
            if block and block in sel.get('blocks', {}):
                test_codes = list(sel['blocks'][block].get('test', []))
        all_codes = sorted(set(train_codes) | set(test_codes))
        prefix_vocab, code_to_prefix_ids = build_prefix_vocab(all_codes)
        print(f"  Prefix vocab size: {len(prefix_vocab)}", flush=True)
        print(f"  Codes covered:     {len(all_codes)} (train={len(train_codes)}, test={len(test_codes)})", flush=True)
        return ICLTransformerV11FeatAttnDiseaseEmb(
            prefix_vocab=prefix_vocab,
            code_to_prefix_ids=code_to_prefix_ids,
            **common,
            per_prot_hidden=model_cfg.get('per_prot_hidden', 32),
            per_prot_heads =model_cfg.get('per_prot_heads', 4),
            per_prot_layers=model_cfg.get('per_prot_layers', 3),
            n_cls_tokens   =model_cfg.get('n_cls_tokens', 4),
            n_cls_feat     =model_cfg.get('n_cls_feat', 24),
            n_feat_layers  =model_cfg.get('n_feat_layers', 1),
            feat_heads     =model_cfg.get('feat_heads', 4),
        ), {'prefix_vocab': prefix_vocab, 'code_to_prefix_ids': code_to_prefix_ids}

    raise ValueError(f"Unknown model variant: {variant!r}")


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

    model_cfg = config['model']
    model, extras = build_model(model_cfg, config, processed_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel variant: {model_cfg.get('variant', 'v6')}", flush=True)
    print(f"  Parameters:  {n_params:,}", flush=True)

    save_dir = os.path.join(project_root, config['training']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)

    # For diseaseemb, persist the prefix vocab so eval can rebuild the model.
    if extras is not None:
        with open(os.path.join(save_dir, 'prefix_vocab.pkl'), 'wb') as f:
            pickle.dump(extras, f)
        print(f"  Saved prefix_vocab.pkl to {save_dir}", flush=True)

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
