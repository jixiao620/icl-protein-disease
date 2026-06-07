#!/usr/bin/env python -u
"""
Generic cross-block evaluation script.
Usage:
  python scripts/eval_crossblock.py \
    --checkpoint checkpoints_full_gpt/best_model.pt \
    --data_dir processed_data_c \
    --diseases C34 C18 C43 C53 C54 \
    --output results/i_model_on_c.json \
    --label "I model -> C test"
"""
import sys, os, argparse, json, pickle
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--diseases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--context_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = '/work/jl1401/icl_protein_disease'

    print(f"\n{'='*60}")
    print(f"Cross-block Evaluation: {args.label}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Diseases:   {args.diseases}")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n", flush=True)

    # Load test data
    data_path = os.path.join(project_root, args.data_dir, 'test_data.pkl')
    if not os.path.exists(data_path):
        # try train_data.pkl as fallback (for cross-block, disease may be in train set of other block)
        data_path = os.path.join(project_root, args.data_dir, 'train_data.pkl')
    print(f"Loading data from: {data_path}", flush=True)
    with open(data_path, 'rb') as f:
        base_data = pickle.load(f)

    # Also try the other split file
    other_path = data_path.replace('test_data', 'train_data') if 'test_data' in data_path else data_path.replace('train_data', 'test_data')
    if os.path.exists(other_path):
        with open(other_path, 'rb') as f:
            other_data = pickle.load(f)
        base_data.update(other_data)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = InContextTransformerGPT(
        protein_dim=2941, hidden_dim=768, n_layers=12, n_heads=12,
        dropout=0.2, temperature=1.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded.", flush=True)

    results = {}
    for disease in args.diseases:
        if disease not in base_data:
            print(f"\nWARNING: {disease} not found in data, skipping.", flush=True)
            results[disease] = {'error': 'disease not found in data'}
            continue

        print(f"\n{'='*60}\n{disease}", flush=True)
        X, y = base_data[disease]
        print(f"Samples: {len(y)}, Positives: {int(y.sum())} ({y.mean()*100:.2f}%)", flush=True)

        if y.sum() < 5:
            print(f"WARNING: too few positives, skipping.", flush=True)
            results[disease] = {'error': 'too few positives', 'n_samples': len(y), 'n_positives': int(y.sum())}
            continue

        dataset = InContextDiseaseDatasetNoID(
            data_dict={disease: (X, y)},
            context_size=args.context_size,
            is_training=False
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn_no_id, num_workers=0)

        all_probs, all_targets = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i % 200 == 0:
                    print(f"  Batch {i}/{len(loader)}", flush=True)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(batch['target_label'].cpu().numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        preds = (all_probs > 0.5).astype(int)

        auroc  = float(roc_auc_score(all_targets, all_probs))
        auprc  = float(average_precision_score(all_targets, all_probs))
        acc    = float(accuracy_score(all_targets, preds))
        prec   = float(precision_score(all_targets, preds, zero_division=0))
        rec    = float(recall_score(all_targets, preds, zero_division=0))
        f1     = float(f1_score(all_targets, preds, zero_division=0))

        results[disease] = {
            'n_samples': len(all_targets), 'n_positives': int(all_targets.sum()),
            'prevalence': float(all_targets.mean()),
            'auroc': auroc, 'auprc': auprc, 'accuracy': acc,
            'precision': prec, 'recall': rec, 'f1': f1
        }
        print(f"  AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}  F1: {f1:.4f}", flush=True)

    valid = [r['auroc'] for r in results.values() if 'auroc' in r and 'error' not in r]
    summary = {'mean_auroc': float(np.mean(valid)) if valid else None, 'label': args.label}
    results['summary'] = summary
    print(f"\nMean AUROC: {summary['mean_auroc']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)

if __name__ == '__main__':
    main()
