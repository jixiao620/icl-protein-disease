"""
TabPFN-3 multi-task fine-tune on our training-disease pool (per block).

Concept parallel to our ICL training:
  - Our ICL sees (context patients + labels, query patient) sampled from a POOL
    of training diseases in each mini-batch — one disease per batch.
  - Here we do the same: TabPFN's meta-fine-tune loop already samples chunks of
    rows from a dataset and splits them into (context, query). We just need to
    feed it a LIST of training-disease (X, y) so each batch is drawn from one
    disease at a time.

Why a subclass:
  Public FinetunedTabPFNClassifier.fit() assumes X is a single ndarray
  (`train_size = X.shape[0]`, sklearn validation, train/val split). But the
  underlying `get_preprocessed_dataset_chunks` accepts a list of datasets
  (verified at runtime: type hint XType | list[XType]). We override the
  parts of _fit that assume a single array; the rest of the loop is
  task-agnostic and works transparently.

Simplifications vs. stock _fit:
  - No validation split, no early stopping — we run a fixed number of epochs.
    Val eval is expensive and requires a single-array X; skipping it lets us
    stay minimal. If loss curves look bad we can add val later.
  - No DDP (single-GPU training).

Usage:
  python -u scripts/finetune_tabpfn_v3.py --block i
  python -u scripts/finetune_tabpfn_v3.py --block c
  python -u scripts/finetune_tabpfn_v3.py --block g

Output:
  checkpoints_tabpfn_v3_finetuned/<block>/final.ckpt   — fine-tuned TabPFN weights
                                                         (loadable via TabPFNClassifier(model_path=...))
  checkpoints_tabpfn_v3_finetuned/<block>/train_log.json — per-epoch mean loss
"""

import os, sys, json, argparse, time, copy
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetuning import FinetunedTabPFNClassifier
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks, meta_dataset_collator,
)
from tabpfn.finetuning.train_util import (
    get_and_init_optimizer, get_cosine_schedule_with_warmup,
)
from tabpfn.architectures.interface import PerformanceOptions
from tabpfn.utils import infer_random_state


PROJECT  = os.environ.get('ICL_PROJECT_ROOT',
                          '/work/jl1401/icl_release_auto/icl_protein_release')
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT_ROOT = os.path.join(PROJECT, 'checkpoints_tabpfn_v3_finetuned_clean')

# CLEAN pipeline: read training diseases from selection.yaml (auto-picked list)
# and only train on the 70% train-pool patients from patient_split.yaml. Every
# other patient (15% ctx + 15% query) is patient-disjoint from training so
# reviewer-#1 leakage is eliminated end-to-end.
import yaml as _yaml
_SEL   = _yaml.safe_load(open(os.path.join(PROJECT, 'selection.yaml')))
_SPLIT = _yaml.safe_load(open(os.path.join(PROJECT, 'patient_split.yaml')))
TRAIN_DISEASES = {L.lower(): _SEL['blocks'][L]['train']
                  for L in ['I','C','G']
                  if L in _SEL['blocks'] and not _SEL['blocks'][L].get('skipped')}
TRAIN_USERIDS  = set(str(u) for u in _SPLIT['train'])
print(f"[clean] TRAIN_DISEASES = {TRAIN_DISEASES}", flush=True)
print(f"[clean] restricting training to {len(TRAIN_USERIDS)} train-pool patients "
      f"(70% split)", flush=True)

# Cap patients per training disease. Prevents any single (very common) disease
# from dominating the batch mix. Full UKB slice is ~48K; each training disease
# has all patients present with binary labels, so capping via random subsample
# per epoch reduces memory + speeds each epoch.
MAX_PATIENTS_PER_DISEASE = 20000

# Fine-tune hparams (fairly conservative; can retune later)
EPOCHS         = 8       # reduced from 15 — earlier run showed marginal Δloss per epoch
                         # after ep 2 (~0.005/ep); 8 gives ep-5 + ep-8 checkpoints in ~10h
LEARNING_RATE  = 1e-5
WEIGHT_DECAY   = 0.01
GRAD_CLIP      = 1.0
CTX_PLUS_QRY   = 256     # per-batch sample size for meta task
                         # (K_context = CTX_PLUS_QRY * (1 - CTX_QRY_SPLIT) = 128, matches
                         #  our eval K=64/128; larger values OOM on 32GB GPU with 2941 feats)
CTX_QRY_SPLIT  = 0.5     # 50/50 context/query
META_BATCH_SZ  = 1       # how many meta-tasks per gradient step
N_EST_FINETUNE = 15      # must match TabPFN's auto-scaled ensemble size for 2941 features
                         # (auto-scale forces n_estimators to ceil(2941/200)=15 for full feature
                         #  coverage; _forward_with_loss asserts finetune-time E matches this)
MIN_POS = 5              # skip training diseases with fewer positives (trivial signal)
MIN_NEG = 5              # skip training diseases with fewer negatives (e.g., I10 100% positive)
SEED           = 42


# -----------------------------------------------------------------------------
# Data loading (reuse the eval-script helpers)
# -----------------------------------------------------------------------------
def load_protein_data():
    fnames = sorted(f for f in os.listdir(DATA_DIR)
                    if f.startswith('xa') and f.endswith('.gz'))
    dfs, cols = [], None
    for fn in fnames:
        fp = os.path.join(DATA_DIR, fn)
        if cols is None:
            df = pd.read_csv(fp, compression='gzip')
            cols = df.columns.tolist()
        else:
            df = pd.read_csv(fp, compression='gzip', header=None, names=cols)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def get_disease_xy(protein_df, disease_df, code):
    col = next((c for c in disease_df.columns
                if c.startswith(f'Union#{code}#') and c.split('#')[1] == code), None)
    if col is None:
        return None, None
    label_df = disease_df[['userID', col]].dropna()
    # CLEAN: restrict to train-pool patients only (drops 30% held-out)
    label_df = label_df[label_df['userID'].astype(str).isin(TRAIN_USERIDS)]
    merged   = protein_df.merge(label_df, on='userID', how='inner')
    if len(merged) == 0:
        return None, None
    y = merged[col].values.astype(np.int32)
    X = merged.drop(columns=['userID', col]).values.astype(np.float32)
    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = 0.0
    return X, y


def build_training_pool(block):
    print(f"Loading UKB proteomics data...", flush=True)
    protein_df = load_protein_data()
    disease_df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'),
                             compression='gzip')
    print(f"  {protein_df.shape[0]} patients x {protein_df.shape[1]-1} proteins",
          flush=True)

    X_list, y_list, names = [], [], []
    rng = np.random.RandomState(SEED)
    for code in TRAIN_DISEASES[block]:
        X, y = get_disease_xy(protein_df, disease_df, code)
        if X is None:
            print(f"  {code}: NOT FOUND — skipping", flush=True)
            continue
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < MIN_POS or n_neg < MIN_NEG:
            print(f"  {code}: skipping — N+={n_pos}, N-={n_neg} "
                  f"(need ≥{MIN_POS} pos AND ≥{MIN_NEG} neg for meaningful signal)",
                  flush=True)
            continue
        # Subsample to cap per-disease size, capping BOTH classes so we don't
        # accidentally strip all negatives (or all positives) when one class
        # dominates. Cap each class at MAX/2 → balanced-ish subsample.
        if len(y) > MAX_PATIENTS_PER_DISEASE:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos_keep = min(len(pos_idx), MAX_PATIENTS_PER_DISEASE // 2)
            n_neg_keep = min(len(neg_idx), MAX_PATIENTS_PER_DISEASE - n_pos_keep)
            pos_pick = (rng.choice(len(pos_idx), n_pos_keep, replace=False)
                        if n_pos_keep < len(pos_idx) else np.arange(len(pos_idx)))
            neg_pick = (rng.choice(len(neg_idx), n_neg_keep, replace=False)
                        if n_neg_keep < len(neg_idx) else np.arange(len(neg_idx)))
            keep_idx = np.concatenate([pos_idx[pos_pick], neg_idx[neg_pick]])
            keep_idx.sort()
            X = X[keep_idx]; y = y[keep_idx]
        print(f"  {code}: N={len(y)}, N+={int(y.sum())}, N-={int(len(y)-y.sum())}",
              flush=True)
        X_list.append(X); y_list.append(y); names.append(code)

    if len(X_list) == 0:
        raise RuntimeError(f"No usable training diseases for block {block}")
    return X_list, y_list, names


# -----------------------------------------------------------------------------
# Multi-task fine-tune subclass
# -----------------------------------------------------------------------------
class MultiTaskFinetuner(FinetunedTabPFNClassifier):
    """Multi-task variant that accepts LIST of (X_i, y_i) — one per training
    disease. Skips the val/early-stopping logic that assumes a single array."""

    def fit_multi(self, X_list, y_list, output_dir: Path):
        """Reimplemented training loop for list inputs. Modeled on
        FinetunedTabPFNBase._fit but with:
          - X, y as list of ndarray (one per task)
          - No validation split, no early stopping
          - No DDP
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        assert len(X_list) == len(y_list), "X_list, y_list length mismatch"
        train_size = sum(len(y) for y in y_list)     # for checkpoint naming only
        n_features = X_list[0].shape[1]
        print(f"[MT-FT] Total rows across {len(X_list)} tasks: {train_size:,} "
              f"× {n_features} features", flush=True)

        # ── build the underlying TabPFN estimator (mirrors _fit lines 647-703)
        _estimator_kwargs = copy.deepcopy(self._estimator_kwargs)
        model_path = _estimator_kwargs.pop("model_path", None)
        base_estimator_config = {
            **_estimator_kwargs,
            "ignore_pretraining_limits": True,
            "device": self.device,
            "random_state": self.random_state,
        }
        finetuning_estimator_config = self._build_estimator_config(
            base_estimator_config, self.n_estimators_finetune,
        )
        if model_path is not None:
            finetuning_estimator_config["model_path"] = model_path
        self.finetuned_estimator_ = self._create_estimator(finetuning_estimator_config)
        self._setup_estimator()
        self.finetuned_estimator_._initialize_model_variables()
        self.n_features_in_ = n_features
        self.finetuned_estimator_.model_.to(self.device)

        finetuning_performance_options = PerformanceOptions(
            force_recompute_layer=self.use_activation_checkpointing,
            use_chunkwise_inference=False,
        )

        # ── optimizer
        model_for_opt = self.finetuned_estimator_.model_
        optimizer = get_and_init_optimizer(
            model_parameters=model_for_opt.parameters(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            checkpoint_path=None,
            device=self.device,
        )
        use_amp = self.device.startswith("cuda") and torch.cuda.is_available()
        scaler  = GradScaler() if use_amp else None

        static_seed, rng = infer_random_state(self.random_state)
        preprocessing_random_state = (
            static_seed if self.use_fixed_preprocessing_seed else rng
        )

        # ── figure out the query size per batch (uses base-class helper)
        finetuning_query_size = self._get_valid_finetuning_query_size(
            query_size=int(self.n_finetune_ctx_plus_query_samples
                           * self.finetune_ctx_query_split_ratio),
            y_train=y_list[0],           # only used to sanity-check size
        )
        print(f"[MT-FT] batches/epoch will use {self.n_finetune_ctx_plus_query_samples} "
              f"rows/task, query_size={finetuning_query_size}", flush=True)

        scheduler = None
        train_log = []
        for epoch in range(self.epochs):
            epoch_start = time.monotonic()
            epoch_random_state = static_seed + epoch

            training_splitter = partial(
                train_test_split, test_size=finetuning_query_size,
                random_state=epoch_random_state,
            )

            # KEY LINE — pass X_list, y_list (accepted as list per XType|list[XType])
            # force_no_stratify=True: don't try to stratify chunks — some training
            # diseases may have extreme class imbalance where stratified splits fail.
            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=self.finetuned_estimator_,
                X_raw=X_list, y_raw=y_list,
                split_fn=training_splitter,
                max_data_size=self.n_finetune_ctx_plus_query_samples,
                model_type=self._model_type,
                equal_split_size=False,
                data_shuffle_seed=epoch_random_state,
                preprocessing_random_state=preprocessing_random_state,
                force_no_stratify=True,
            )

            dataloader_generator = torch.Generator().manual_seed(epoch_random_state)
            finetuning_dataloader = DataLoader(
                training_datasets,
                batch_size=META_BATCH_SZ,
                collate_fn=meta_dataset_collator,
                shuffle=True,
                generator=dataloader_generator,
            )

            # LR scheduler (only build once)
            if self.use_lr_scheduler and scheduler is None:
                steps_per_epoch = len(finetuning_dataloader)
                if steps_per_epoch == 0:
                    print("[MT-FT] no batches — stopping", flush=True); break
                total_steps  = steps_per_epoch * self.epochs
                warmup_steps = int(total_steps * 0.1)
                lrate_fn = get_cosine_schedule_with_warmup(
                    total_steps=total_steps, warmup_steps=warmup_steps,
                    warmup_only=False,
                )
                scheduler = LambdaLR(optimizer, lr_lambda=lrate_fn)
                print(f"[MT-FT] LR schedule: {total_steps} total steps, "
                      f"{warmup_steps} warmup", flush=True)

            epoch_loss_sum, epoch_batches = 0.0, 0
            pbar = tqdm(finetuning_dataloader,
                        desc=f"[MT-FT ep {epoch+1}/{self.epochs}]",
                        leave=False)
            for batch in pbar:
                optimizer.zero_grad()
                if self._should_skip_batch(batch):
                    continue
                self._setup_batch(batch)
                self.finetuned_estimator_.fit_from_preprocessed(
                    batch.X_context, batch.y_context,
                    batch.cat_indices, batch.configs,
                    performance_options=finetuning_performance_options,
                )
                with autocast(enabled=(use_amp and scaler is not None)):
                    loss = self._forward_with_loss(batch)
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if self.grad_clip_value is not None:
                        clip_grad_norm_(model_for_opt.parameters(),
                                        self.grad_clip_value)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip_value is not None:
                        clip_grad_norm_(model_for_opt.parameters(),
                                        self.grad_clip_value)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                epoch_loss_sum += float(loss.detach().item())
                epoch_batches  += 1
                pbar.set_postfix(loss=f"{float(loss.detach().item()):.4f}")

            mean_loss = epoch_loss_sum / max(1, epoch_batches)
            elapsed   = time.monotonic() - epoch_start
            lr        = scheduler.get_last_lr()[0] if scheduler is not None else self.learning_rate
            log_entry = {"epoch": epoch+1, "batches": epoch_batches,
                         "mean_loss": mean_loss, "lr": lr, "elapsed_s": elapsed}
            train_log.append(log_entry)
            print(f"[MT-FT ep {epoch+1}/{self.epochs}] loss={mean_loss:.4f}  "
                  f"batches={epoch_batches}  lr={lr:.2e}  {elapsed:.0f}s",
                  flush=True)

            # Save intermediate checkpoint every 5 epochs
            # NOTE: tabpfn 8.1.0's save_tabpfn_model() has a bug with TabPFNV3
            # (expects .models_ attr on model_, which TabPFNV3 lacks). Bypass
            # by saving raw state_dict — eval script uses torch.load + swap in.
            if (epoch + 1) % 5 == 0 or epoch + 1 == self.epochs:
                ckpt_path = output_dir / f"epoch_{epoch+1}.pt"
                torch.save(
                    self.finetuned_estimator_.model_.state_dict(),
                    ckpt_path,
                )
                print(f"[MT-FT] saved {ckpt_path}", flush=True)

            with open(output_dir / "train_log.json", "w") as f:
                json.dump(train_log, f, indent=2)

        # Final checkpoint (raw state_dict, see note above)
        final_path = output_dir / "final.pt"
        torch.save(
            self.finetuned_estimator_.model_.state_dict(),
            final_path,
        )
        print(f"[MT-FT] DONE. Final checkpoint: {final_path}", flush=True)
        return self


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=list(TRAIN_DISEASES.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr",     type=float, default=LEARNING_RATE)
    parser.add_argument("--ctx_plus_qry", type=int, default=CTX_PLUS_QRY)
    args = parser.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}", flush=True)
    print(f"Block: {args.block}   Epochs: {args.epochs}   LR: {args.lr}   "
          f"ctx+qry: {args.ctx_plus_qry}", flush=True)

    X_list, y_list, names = build_training_pool(args.block)
    print(f"Training pool: {names}", flush=True)

    out_dir = Path(OUT_ROOT) / args.block

    ft = MultiTaskFinetuner(
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        validation_split_ratio=0.0,
        early_stopping=False,
        n_finetune_ctx_plus_query_samples=args.ctx_plus_qry,
        finetune_ctx_query_split_ratio=CTX_QRY_SPLIT,
        random_state=SEED,
        grad_clip_value=GRAD_CLIP,
        use_lr_scheduler=True,
        n_estimators_finetune=N_EST_FINETUNE,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        use_activation_checkpointing=True,    # safer — 32GB GPU can OOM on K=128 + 15 ensemble
        use_fixed_preprocessing_seed=True,
    )
    ft.fit_multi(X_list, y_list, output_dir=out_dir)


if __name__ == "__main__":
    main()
