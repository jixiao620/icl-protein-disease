#!/usr/bin/env python
"""
One-shot architecture + data diagnostic.
Answers: NaN presence, PE type, double-PE, query label embedding, baseline scores.
"""
import pickle, sys, os
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PROJECT = '/work/jl1401/icl_protein_disease'

# ─── 1. NaN check ────────────────────────────────────────────────
print("=" * 60)
print("1. NaN CHECK")
print("=" * 60)
for split_dir in ['processed_data_c', 'processed_data_g']:
    for fname in ['train_data.pkl', 'test_data.pkl']:
        path = os.path.join(PROJECT, split_dir, fname)
        with open(path, 'rb') as f:
            d = pickle.load(f)
        total_vals, total_nan = 0, 0
        for code, (X, y) in d.items():
            n = np.isnan(X).sum()
            total_nan += n
            total_vals += X.size
        pct = total_nan / total_vals * 100
        print(f"  {split_dir}/{fname}: NaN={total_nan}/{total_vals} ({pct:.4f}%)")

# ─── 2. Load model ───────────────────────────────────────────────
print()
print("=" * 60)
print("2. POSITIONAL ENCODING DIAGNOSIS")
print("=" * 60)
from models.transformer_gpt import InContextTransformerGPT
ckpt = torch.load(os.path.join(PROJECT, 'checkpoints_g_model_v5/best_model.pt'),
                  map_location='cpu', weights_only=False)
model = InContextTransformerGPT(protein_dim=2941, hidden_dim=768,
                                n_layers=12, n_heads=12, dropout=0.2, temperature=1.0)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# ① custom pos_encoding
pe = model.pos_encoding
is_param = any(pe is p for p in model.parameters())
is_buffer = any(pe is b for b in model.buffers())
print(f"  pos_encoding shape: {pe.shape}")
print(f"  is nn.Parameter (trained): {is_param}")
print(f"  is buffer (fixed):         {is_buffer}")
print(f"  pos_encoding[:, :4, :3]:\n{pe[0, :4, :3]}")
print(f"  all zeros? {pe.abs().max().item() < 1e-9}")
print(f"  looks sinusoidal? max={pe.max().item():.4f}  min={pe.min().item():.4f}")

# ② gpt.wpe
wpe = model.gpt.wpe.weight   # [1024, 768]
print(f"\n  gpt.wpe shape: {wpe.shape}")
print(f"  gpt.wpe is trainable param: {wpe.requires_grad}")
print(f"  gpt.wpe[:4, :3]:\n{wpe[:4, :3].data}")
print(f"  wpe all zeros? {wpe.data.abs().max().item() < 1e-9}")

# ③ Is gpt a HuggingFace GPT2Model? (adds wpe internally in forward)
gpt_type = type(model.gpt).__name__
gpt_module = type(model.gpt).__module__
print(f"\n  model.gpt type: {gpt_type}  from {gpt_module}")
hf_gpt2 = 'transformers' in gpt_module
print(f"  Is HuggingFace GPT2Model: {hf_gpt2}")
if hf_gpt2:
    print("  *** WARNING: HF GPT2Model adds wpe internally → double PE! ***")

# ─── 3. Query label embedding ────────────────────────────────────
print()
print("=" * 60)
print("3. QUERY LABEL EMBEDDING")
print("=" * 60)
label_emb = model.label_emb.weight   # [2, 768]
print(f"  label_emb shape: {label_emb.shape}")
print(f"  label_emb[0] (used for query?): norm={label_emb[0].norm().item():.4f}  first 5: {label_emb[0, :5].data}")
print(f"  label_emb[1]: norm={label_emb[1].norm().item():.4f}")
# Check if label_emb[0] is zero (would mean query gets zero vec, safe)
print(f"  label_emb[0] all zeros? {label_emb[0].abs().max().item() < 1e-9}")

# Try to check forward pass source for query label handling
import inspect
try:
    src = inspect.getsource(model.forward)
    # Look for query label handling
    lines = [l for l in src.split('\n') if 'query' in l.lower() or 'label_emb' in l.lower()]
    print("\n  Forward source lines mentioning query/label_emb:")
    for l in lines[:20]:
        print(f"    {l}")
except Exception as e:
    print(f"  Cannot get forward source: {e}")

# ─── 4. Baseline per-patient scores ──────────────────────────────
print()
print("=" * 60)
print("4. BASELINE PER-PATIENT SCORES")
print("=" * 60)
import json
results_dir = os.path.join(PROJECT, 'results')
for fname in ['baseline_xgboost_c.json', 'baseline_dnn_c.json',
              'baseline_xgboost_g.json', 'baseline_dnn_g.json']:
    path = os.path.join(results_dir, fname)
    with open(path) as f:
        d = json.load(f)
    # Check if any key has per-patient scores
    sample = list(d.values())[0] if d else {}
    keys = list(sample.keys()) if isinstance(sample, dict) else []
    has_scores = any(k in keys for k in ['scores', 'probs', 'predictions', 'y_pred', 'y_score'])
    print(f"  {fname}: keys={keys[:8]}  has_scores={has_scores}")

# ─── 5. Test split overlap ───────────────────────────────────────
print()
print("=" * 60)
print("5. TRAIN/TEST SPLIT CONSISTENCY")
print("=" * 60)
# ICL uses processed_data_c/test_data.pkl
# Baseline uses same raw data split the same way?
# Check n_samples match
for block in ['c', 'g']:
    pkl = os.path.join(PROJECT, f'processed_data_{block}/test_data.pkl')
    with open(pkl, 'rb') as f:
        d = pickle.load(f)
    for code, (X, y) in d.items():
        print(f"  ICL test {code}: N={len(y)}, N+={int(y.sum())}")

print()
print("=== DONE ===")
