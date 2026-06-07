#!/usr/bin/env python -u
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import pickle
import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60, flush=True)
print("🧪 Experiment 3: I Model → O Rare Diseases", flush=True)
print("="*60, flush=True)
print("Purpose: Prove cross-domain transfer fails", flush=True)
print("Expected: AUROC ~0.6-0.7 (near random)", flush=True)
print("="*60, flush=True)

# Load I model (trained on I10, I12, I120 with context=32)
print("\nLoading I model...", flush=True)
checkpoint = torch.load('checkpoints_full_gpt/best_model.pt', map_location=device)
model = InContextTransformerGPT(
    protein_dim=2941,
    hidden_dim=768,
    n_layers=12,
    n_heads=12,
    dropout=0.2,
    temperature=1.0
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"✓ I Model loaded (trained on I10/I12/I120, context=32)", flush=True)

# Load O rare diseases test data
print("\nLoading O rare diseases test data...", flush=True)
with open('processed_data_o/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Select only 4 rare diseases for testing
all_diseases = list(test_data.keys())
o_rare_diseases = all_diseases[:4]  # Take first 4
print(f"Testing on {len(o_rare_diseases)} rare diseases: {o_rare_diseases}", flush=True)

results = {}

# Evaluate on each O rare disease
for disease in o_rare_diseases:
    print(f"\n{'='*60}", flush=True)
    print(f"{disease}", flush=True)
    print(f"{'='*60}", flush=True)
    
    X, y = test_data[disease]
    n_pos = int(y.sum())
    prev = y.mean() * 100
    
    print(f"Samples: {len(y)}, Positives: {n_pos} ({prev:.4f}%)", flush=True)
    
    # Create dataset with context=64
    dataset = InContextDiseaseDatasetNoID(
        data_dict={disease: (X, y)},
        context_size=64,
        is_training=False
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, 
                       collate_fn=collate_fn_no_id, num_workers=0)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 500 == 0:
                print(f"  Batch {i}/{len(loader)}", flush=True)
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(batch['target_label'].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    auroc = roc_auc_score(all_targets, all_probs)
    auprc = average_precision_score(all_targets, all_probs)
    
    all_preds = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print(f"  AUROC:     {auroc:.4f}", flush=True)
    print(f"  AUPRC:     {auprc:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  F1:        {f1:.4f}", flush=True)
    
    results[disease] = {
        'n_samples': len(y),
        'n_pos': n_pos,
        'prevalence': prev,
        'auroc': float(auroc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }

# Summary
print(f"\n{'='*60}", flush=True)
print("SUMMARY: I Model → O Rare Diseases", flush=True)
print(f"{'='*60}", flush=True)

aurocs = [v['auroc'] for v in results.values()]
auprcs = [v['auprc'] for v in results.values()]
recalls = [v['recall'] for v in results.values()]

print(f"Mean AUROC:  {np.mean(aurocs):.4f}", flush=True)
print(f"Mean AUPRC:  {np.mean(auprcs):.4f}", flush=True)
print(f"Mean Recall: {np.mean(recalls):.4f}", flush=True)

results['summary'] = {
    'mean_auroc': float(np.mean(aurocs)),
    'mean_auprc': float(np.mean(auprcs)),
    'mean_recall': float(np.mean(recalls))
}

# Save
Path('results').mkdir(exist_ok=True)
with open('results/i_model_on_o_rare.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved to results/i_model_on_o_rare.json", flush=True)
