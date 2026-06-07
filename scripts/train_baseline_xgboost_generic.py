#!/usr/bin/env python -u
"""Generic XGBoost baseline."""
import sys, os, argparse, json, pickle
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--diseases', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--block', default='')
    args = parser.parse_args()

    project_root = '/work/jl1401/icl_protein_disease'
    print(f"\n{'='*60}\nXGBoost Baseline: {args.block} block\n{'='*60}\n", flush=True)

    all_data = {}
    for split in ['train_data.pkl', 'test_data.pkl']:
        p = os.path.join(project_root, args.data_dir, split)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                all_data.update(pickle.load(f))

    results = {}
    for disease in args.diseases:
        print(f"\n{disease}", flush=True)
        if disease not in all_data:
            results[disease] = {'error': 'not found'}
            continue

        X, y = all_data[disease]
        print(f"Samples: {len(y)}, Positives: {int(y.sum())}", flush=True)

        if y.sum() < 5:
            results[disease] = {'error': 'insufficient positives'}
            continue

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scale_pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.01,
            scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs > 0.5).astype(int)

        try: auroc = float(roc_auc_score(y_te, probs))
        except: auroc = None
        try: auprc = float(average_precision_score(y_te, probs))
        except: auprc = None

        results[disease] = {
            'n_total': len(y), 'n_test': len(y_te), 'n_test_pos': int(y_te.sum()),
            'auroc': auroc, 'auprc': auprc,
            'accuracy': float(accuracy_score(y_te, preds)),
            'precision': float(precision_score(y_te, preds, zero_division=0)),
            'recall': float(recall_score(y_te, preds, zero_division=0)),
            'f1': float(f1_score(y_te, preds, zero_division=0))
        }
        print(f"  AUROC: {auroc}  AUPRC: {auprc}", flush=True)

    valid = [r['auroc'] for r in results.values() if r.get('auroc') is not None]
    results['summary'] = {'mean_auroc': float(np.mean(valid)) if valid else None, 'block': args.block}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}", flush=True)

if __name__ == '__main__':
    main()
