
"""

Training Loop for In-Context Transformer

"""



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader

import os

import time

from typing import Dict

import numpy as np

from sklearn.metrics import roc_auc_score




class Trainer:

    """Trainer for in-context disease prediction"""



    def __init__(

        self,

        model: nn.Module,

        train_loader: DataLoader,

        val_loader: DataLoader,

        config: dict,

        device: torch.device,

        save_dir: str = 'checkpoints'

    ):

        self.model = model.to(device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.config = config

        self.device = device

        self.save_dir = save_dir



        os.makedirs(save_dir, exist_ok=True)



        # Optimizer

        self.optimizer = optim.Adam(

            model.parameters(),

            lr=config['training']['learning_rate'],

            weight_decay=config['training']['weight_decay']

        )



        # Learning rate scheduler

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(

            self.optimizer,

            T_max=config['training']['num_epochs'],

            eta_min=0.000001

        )



        # Loss function (weighted BCE for class imbalance)

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')



        # Tracking — checkpoint criterion is val AUROC (higher = better)

        self.best_val_auroc = 0.0

        self.patience_counter = 0

        self.train_losses = []

        self.val_losses = []



    def compute_loss(self, batch, class_weights=None):

        """Compute weighted loss"""

        logits = self.model(batch)

        targets = batch['target_label']



        # BCE loss

        loss = self.criterion(logits, targets)



        # Apply class weights to positive samples only
        if class_weights is not None:
            weights = torch.ones_like(loss)
            for disease_code, pos_weight in class_weights.items():
                disease_mask = torch.tensor([
                    code == disease_code
                    for code in batch['query_disease_codes']
                ], device=self.device)
                pos_mask = disease_mask & (batch['target_label'] == 1)
                weights[pos_mask] = pos_weight

            loss = (loss * weights).mean()
        else:
            loss = loss.mean()



        return loss, logits



    def train_epoch(self, epoch, class_weights=None):

        """Train for one epoch"""

        self.model.train()

        total_loss = 0

        n_batches = 0



        for batch_idx, batch in enumerate(self.train_loader):

            # Move to device

            batch = {

                k: v.to(self.device) if isinstance(v, torch.Tensor) else v

                for k, v in batch.items()

            }



            # Forward

            loss, logits = self.compute_loss(batch, class_weights)



            # Backward

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()



            total_loss += loss.item()

            n_batches += 1



            # Log

            if batch_idx % self.config['training']['log_interval'] == 0:

                print(f"   Batch {batch_idx}/{len(self.train_loader)}, "

                      f"Loss: {loss.item():.4f}")



        avg_loss = total_loss / n_batches

        return avg_loss



    @torch.no_grad()

    def validate(self, class_weights=None):

        """Validate — returns (val_loss, val_acc, val_auroc)"""

        self.model.eval()

        total_loss = 0

        n_batches = 0



        all_preds = []

        all_targets = []



        for batch in self.val_loader:

            batch = {

                k: v.to(self.device) if isinstance(v, torch.Tensor) else v

                for k, v in batch.items()

            }



            loss, logits = self.compute_loss(batch, class_weights)



            total_loss += loss.item()

            n_batches += 1



            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())

            all_targets.append(batch['target_label'].cpu().numpy())



        avg_loss = total_loss / n_batches



        all_preds = np.concatenate(all_preds)

        all_targets = np.concatenate(all_targets)



        # Compute accuracy

        acc = ((all_preds > 0.5) == all_targets).mean()



        # Compute AUROC (falls back to 0.5 if only one class present in val batch)

        try:

            auroc = roc_auc_score(all_targets, all_preds)

        except ValueError:

            auroc = 0.5



        return avg_loss, acc, auroc



    def train(self, class_weights=None):

        """Full training loop"""

        print("\n" + "="*60)

        print("Starting Training")

        print("="*60)



        for epoch in range(self.config['training']['num_epochs']):

            epoch_start = time.time()



            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")

            print("-" * 60)



            # Train

            train_loss = self.train_epoch(epoch, class_weights)



            # Validate

            if (epoch + 1) % self.config['training']['eval_interval'] == 0:

                val_loss, val_acc, val_auroc = self.validate(class_weights)



                self.train_losses.append(train_loss)

                self.val_losses.append(val_loss)



                # Learning rate step

                self.scheduler.step()



                epoch_time = time.time() - epoch_start



                print(f"\n   Train Loss: {train_loss:.4f}")

                print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUROC: {val_auroc:.4f}")

                print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                print(f"   Time: {epoch_time:.1f}s")



                # Save best model by val AUROC

                if val_auroc > self.best_val_auroc:

                    self.best_val_auroc = val_auroc

                    self.patience_counter = 0

                    self.save_checkpoint('best_model.pt', epoch, val_auroc)

                    print(f"   Saved best model (AUROC: {val_auroc:.4f})")

                else:

                    self.patience_counter += 1



                # Early stopping

                if self.patience_counter >= self.config['training']['early_stopping_patience']:

                    print(f"\n  Early stopping triggered after {epoch+1} epochs")

                    break



            # Save checkpoint periodically

            if (epoch + 1) % 10 == 0:

                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, train_loss)



        # Save final model

        self.save_checkpoint('last_model.pt', epoch, train_loss)



        print("\n" + "="*60)

        print("Training Complete!")

        print(f"   Best Val AUROC: {self.best_val_auroc:.4f}")

        print("="*60)



    def save_checkpoint(self, filename, epoch, metric):

        """Save model checkpoint"""

        checkpoint = {

            'epoch': epoch,

            'model_state_dict': self.model.state_dict(),

            'optimizer_state_dict': self.optimizer.state_dict(),

            'metric': metric,

            'config': self.config

        }

        path = os.path.join(self.save_dir, filename)

        torch.save(checkpoint, path)
