"""
Evaluate ablation checkpoints on all 5 C test diseases.
Saves per-patient scores to .npz for bootstrap CI.
"""

import sys, os, argparse, json, pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id

sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

PROJECT = '/work/jl1401/icl_protein_disease'
TEST_DISEASES = ['C34', 'C18', 'C43', 'C53', 'C54']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label',      required=True,
                        help='condition name, e.g. drop_C44')
    parser.add_argument('--output_json', required=True)
    parser.add_argument('--output_npz',  required=True,
                        help='.npz path for per-patient scores')
    parser.add_argument('--context_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Condition: {args.label}  checkpoint: {args.checkpoint}  device: {device}")

    # Load test data (merge train+test pkl so all diseases are accessible)
    with open(os.path.join(PROJECT, 'processed_data_c/test_data.pkl'), 'rb') as f:
        base_data = pickle.load(f)
    with open(os.path.join(PROJECT, 'processed_data_c/train_data.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    base_data.update(train_data)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = InContextTransformerGPT(
        protein_dim=2941, hidden_dim=768, n_layers=12, n_heads=12,
        dropout=0.2, temperature=1.0
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded.")

    results = {}
    all_scores = {}   # disease → {'probs': array, 'targets': array}

    for disease in TEST_DISEASES:
        if disease not in base_data:
            print(f"WARNING: {disease} not found, skipping.")
            continue

        X, y = base_data[disease]
        print(f"\n{disease}: N={len(y)}, N+={int(y.sum())} ({y.mean()*100:.2f}%)")

        dataset = InContextDiseaseDatasetNoID(
            data_dict={disease: (X, y)},
            context_size=args.context_size,
            is_training=False,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False,
                            collate_fn=collate_fn_no_id, num_workers=0)

        probs, targets = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i % 300 == 0:
                    print(f"  batch {i}/{len(loader)}", flush=True)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                logits = model(batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                targets.extend(batch['target_label'].cpu().numpy().tolist())

        probs   = np.array(probs,   dtype=np.float32)
        targets = np.array(targets, dtype=np.int32)

        auroc = float(roc_auc_score(targets, probs))
        auprc = float(average_precision_score(targets, probs))
        print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

        results[disease] = {
            'n_samples':   int(len(targets)),
            'n_positives': int(targets.sum()),
            'auroc':       auroc,
            'auprc':       auprc,
        }
        all_scores[disease] = {'probs': probs, 'targets': targets}

    results['summary'] = {
        'mean_auroc': float(np.mean([v['auroc'] for v in results.values()
                                     if isinstance(v, dict) and 'auroc' in v])),
        'label': args.label,
    }

    print(f"\n=== SUMMARY ({args.label}) ===")
    for d in TEST_DISEASES:
        if d in results:
            print(f"  {d}: AUROC={results[d]['auroc']:.4f}")
    print(f"  mean: {results['summary']['mean_auroc']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Save per-patient scores
    npz_data = {}
    for disease, sc in all_scores.items():
        npz_data[f'{disease}_probs']   = sc['probs']
        npz_data[f'{disease}_targets'] = sc['targets']
    np.savez_compressed(args.output_npz, **npz_data)
    print(f"\nSaved: {args.output_json}  {args.output_npz}")


if __name__ == '__main__':
    main()
