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
print("📊 Context=64 Model: Full Metrics Evaluation", flush=True)
print("="*60, flush=True)

# Load model
print("Loading model...", flush=True)
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
print(f"✓ Model loaded (device: {device})", flush=True)

# Load test data
with open('processed_data_with_q/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Test diseases
i_diseases = ['I11', 'I119', 'I129', 'I15', 'I110', 'I13', 'I151']
q_diseases = ['Q210', 'Q273', 'Q62', 'Q658', 'Q610']

results = {
    'i_diseases': {},
    'q_diseases': {},
    'context_size': 64
}

# Evaluate I diseases
print("\n" + "="*60, flush=True)
print("POSITIVE CONTROL: I Diseases", flush=True)
print("="*60, flush=True)

for disease in i_diseases:
    print(f"\n{'='*60}", flush=True)
    print(f"{disease}", flush=True)
    print(f"{'='*60}", flush=True)
    
    X, y = test_data[disease]
    n_pos = int(y.sum())
    prev = y.mean() * 100
    
    print(f"Samples: {len(y)}, Positives: {n_pos} ({prev:.2f}%)", flush=True)
    
    # Create dataset with context=64
    dataset = InContextDiseaseDatasetNoID(
        data_dict={disease: (X, y)},
        context_size=64,  # ← Using 64
        is_training=False
    )
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False, 
                       collate_fn=collate_fn_no_id, num_workers=0)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 200 == 0:
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
    
    # Binary predictions (threshold 0.5)
    all_preds = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print(f"  AUROC:     {auroc:.4f}", flush=True)
    print(f"  AUPRC:     {auprc:.4f}", flush=True)
    print(f"  Accuracy:  {accuracy:.4f}", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  F1:        {f1:.4f}", flush=True)
    
    results['i_diseases'][disease] = {
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

# Evaluate Q diseases
print("\n" + "="*60, flush=True)
print("NEGATIVE CONTROL: Q Diseases", flush=True)
print("="*60, flush=True)

for disease in q_diseases:
    print(f"\n{'='*60}", flush=True)
    print(f"{disease}", flush=True)
    print(f"{'='*60}", flush=True)
    
    X, y = test_data[disease]
    n_pos = int(y.sum())
    prev = y.mean() * 100
    
    print(f"Samples: {len(y)}, Positives: {n_pos} ({prev:.2f}%)", flush=True)
    
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
            if i % 200 == 0:
                print(f"  Batch {i}/{len(loader)}", flush=True)
            
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(batch['target_label'].cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    auroc = roc_auc_score(all_targets, all_probs)
    auprc = average_precision_score(all_targets, all_probs)
    
    all_preds = (all_probs > 0.5).astype(int)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    print(f"  AUROC:     {auroc:.4f}", flush=True)
    print(f"  AUPRC:     {auprc:.4f}", flush=True)
    print(f"  Accuracy:  {accuracy:.4f}", flush=True)
    print(f"  Precision: {precision:.4f}", flush=True)
    print(f"  Recall:    {recall:.4f}", flush=True)
    print(f"  F1:        {f1:.4f}", flush=True)
    
    results['q_diseases'][disease] = {
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
print("\n" + "="*60, flush=True)
print("SUMMARY (Context=64)", flush=True)
print("="*60, flush=True)

i_aurocs = [v['auroc'] for v in results['i_diseases'].values()]
i_auprcs = [v['auprc'] for v in results['i_diseases'].values()]
i_recalls = [v['recall'] for v in results['i_diseases'].values()]
i_precisions = [v['precision'] for v in results['i_diseases'].values()]
i_f1s = [v['f1'] for v in results['i_diseases'].values()]

q_aurocs = [v['auroc'] for v in results['q_diseases'].values()]
q_auprcs = [v['auprc'] for v in results['q_diseases'].values()]
q_recalls = [v['recall'] for v in results['q_diseases'].values()]

print(f"\nI Diseases (Positive Control):", flush=True)
print(f"  Mean AUROC:     {np.mean(i_aurocs):.4f}", flush=True)
print(f"  Mean AUPRC:     {np.mean(i_auprcs):.4f}", flush=True)
print(f"  Mean Recall:    {np.mean(i_recalls):.4f}", flush=True)
print(f"  Mean Precision: {np.mean(i_precisions):.4f}", flush=True)
print(f"  Mean F1:        {np.mean(i_f1s):.4f}", flush=True)

print(f"\nQ Diseases (Negative Control):", flush=True)
print(f"  Mean AUROC:  {np.mean(q_aurocs):.4f}", flush=True)
print(f"  Mean AUPRC:  {np.mean(q_auprcs):.4f}", flush=True)
print(f"  Mean Recall: {np.mean(q_recalls):.4f}", flush=True)

results['summary'] = {
    'i_diseases': {
        'mean_auroc': float(np.mean(i_aurocs)),
        'mean_auprc': float(np.mean(i_auprcs)),
        'mean_recall': float(np.mean(i_recalls)),
        'mean_precision': float(np.mean(i_precisions)),
        'mean_f1': float(np.mean(i_f1s))
    },
    'q_diseases': {
        'mean_auroc': float(np.mean(q_aurocs)),
        'mean_auprc': float(np.mean(q_auprcs)),
        'mean_recall': float(np.mean(q_recalls))
    }
}

# Save results
with open('results/ctx64_full_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved to results/ctx64_full_metrics.json", flush=True)
