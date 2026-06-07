#!/usr/bin/env python -u
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import pickle
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.transformer_gpt import InContextTransformerGPT
from data.dataset_no_id import InContextDiseaseDatasetNoID, collate_fn_no_id
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60, flush=True)
print("🧪 Experiment 4: Training O Model", flush=True)
print("="*60, flush=True)
print("Training on: O80 + O70", flush=True)
print("Config: Same as I model (context=32, GPT2)", flush=True)
print("="*60, flush=True)

# Load O training data
print("\nLoading O training data...", flush=True)
with open('processed_data_o/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

print(f"\nTraining diseases:", flush=True)
for disease, (X, y) in train_data.items():
    n_pos = int(y.sum())
    prev = y.mean() * 100
    print(f"  {disease}: {len(y)} samples, {n_pos} positives ({prev:.2f}%)", flush=True)

# Compute class weights (log-flattened, like I model)
class_weights = {}
for disease, (X, y) in train_data.items():
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = n_neg / n_pos
    log_pos_weight = np.log(pos_weight + 1)
    class_weights[disease] = log_pos_weight
    print(f"  {disease} weight: {log_pos_weight:.4f}", flush=True)

# Create dataset (without class_weights parameter)
print("\nCreating dataset...", flush=True)
dataset = InContextDiseaseDatasetNoID(
    data_dict=train_data,
    context_size=32,
    is_training=True
)

print(f"Total samples: {len(dataset)}", flush=True)

# Train/val split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                          collate_fn=collate_fn_no_id, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn_no_id, num_workers=4)

print(f"Train: {train_size}, Val: {val_size}", flush=True)

# Model (same architecture as I model)
print("\nInitializing model...", flush=True)
model = InContextTransformerGPT(
    protein_dim=2941,
    hidden_dim=768,
    n_layers=12,
    n_heads=12,
    dropout=0.2,
    temperature=1.0
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}", flush=True)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

best_val_loss = float('inf')
patience_counter = 0
patience = 15

Path('checkpoints_o_model').mkdir(exist_ok=True)

print("\n" + "="*60, flush=True)
print("🚀 Starting Training", flush=True)
print("="*60, flush=True)

for epoch in range(30):
    print(f"\nEpoch {epoch+1}/30", flush=True)
    print("-"*60, flush=True)
    
    # Training
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch['target_label'].float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        if i % 10 == 0:
            print(f"   Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}", flush=True)
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            logits = model(batch)
            loss = criterion(logits, batch['target_label'].float())
            val_loss += loss.item()
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds == batch['target_label']).sum().item()
            val_total += len(batch['target_label'])
    
    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    print(f"   Train Loss: {train_loss:.4f}", flush=True)
    print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", flush=True)
    print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
    
    scheduler.step()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, 'checkpoints_o_model/best_model.pt')
        print(f"   ✓ Saved best model (val_loss: {val_loss:.4f})", flush=True)
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
        break
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoints_o_model/checkpoint_epoch_{epoch+1}.pt')

# Save last model
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoints_o_model/last_model.pt')

print("\n" + "="*60, flush=True)
print("✅ Training Complete!", flush=True)
print(f"   Best Val Loss: {best_val_loss:.4f}", flush=True)
print("="*60, flush=True)
print(f"📁 Checkpoints: {Path('checkpoints_o_model').absolute()}", flush=True)

