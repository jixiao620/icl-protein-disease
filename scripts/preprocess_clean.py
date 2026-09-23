"""
Patient-disjoint preprocessing for the CLEAN pipeline.

Reads:
  patient_split.yaml (from split_patients.py) — the 3-way userID partition
  selection.yaml     (from auto_select_diseases.py) — per-block train / test codes
  data/data/Original_extracted/Original/data/  — raw UKB-PPP protein + labels

Writes (per block letter):
  processed_data_clean_{letter}/
     train_data.pkl     { training_disease : (X, y) }   TRAIN patients only
     context_data.pkl   { test_disease : (X, y) }       CTX  patients only
     query_data.pkl     { test_disease : (X, y) }       QUERY patients only

Semantics:
  * train_data.pkl fuels train_v6new.py (unchanged) — the ICL model only ever
    sees training-disease labels of patients in the TRAIN pool.
  * context_data.pkl fuels eval_clean.py — at inference the model samples
    K in-context examples from this pool ONLY.
  * query_data.pkl fuels eval_clean.py — AUROC / AUPRC / Brier / ECE are
    computed on these patients ONLY.
  * TRAIN, CTX, QUERY pools are patient-disjoint (guaranteed by
    split_patients.py) so no participant ever crosses the training /
    inference boundary — reviewer concern #1 is thereby closed.

Usage (single block, driven by clean_pipeline.py):
  python scripts/preprocess_clean.py \\
      --block-letter I \\
      --selection selection.yaml \\
      --patient-split patient_split.yaml

Or process all blocks in one call:
  python scripts/preprocess_clean.py \\
      --all --selection selection.yaml --patient-split patient_split.yaml
"""
import argparse
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = os.environ.get("ICL_PROJECT_ROOT",
                              str(Path(__file__).resolve().parent.parent))
DATA_DIR   = os.path.join(PROJECT_ROOT, "data/data/Original_extracted/Original/data")
BINARY_CSV = os.path.join(DATA_DIR, "binary_csv.gz")


def load_protein_data(data_dir):
    print(f"Loading protein files from {data_dir}", flush=True)
    fnames = sorted(f for f in os.listdir(data_dir)
                    if f.startswith("xa") and f.endswith(".gz"))
    if not fnames:
        raise RuntimeError(f"No xa*.gz protein files found in {data_dir}")
    dfs, cols = [], None
    for fn in fnames:
        fp = os.path.join(data_dir, fn)
        if cols is None:
            df = pd.read_csv(fp, compression="gzip"); cols = df.columns.tolist()
        else:
            df = pd.read_csv(fp, compression="gzip", header=None, names=cols)
        dfs.append(df)
        print(f"  {fn}: {len(df)} rows", flush=True)
    return pd.concat(dfs, ignore_index=True)


def find_disease_column(disease_df, code):
    return next((c for c in disease_df.columns
                 if c.startswith(f"Union#{code}#") and c.split("#")[1] == code),
                None)


def extract_disease_split(code, disease_df, protein_df, keep_userids, min_pos=1):
    """Return (X, y) for `code` restricted to userIDs in `keep_userids`.
       min_pos filters diseases that become useless in this pool."""
    col = find_disease_column(disease_df, code)
    if col is None:
        print(f"  {code}: column not found, skipping", flush=True); return None
    label_df = disease_df[["userID", col]].dropna()
    label_df = label_df[label_df["userID"].astype(str).isin(keep_userids)]
    merged   = protein_df.merge(label_df, on="userID", how="inner")
    if len(merged) == 0:
        print(f"  {code}: no matching patients in this pool", flush=True); return None
    y = merged[col].values.astype(np.float32)
    X = merged.drop(columns=["userID", col]).values.astype(np.float32)
    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = 0.0
    n_pos = int(y.sum())
    if n_pos < min_pos:
        print(f"  {code}: only {n_pos} positives (< {min_pos}), skipping",
              flush=True); return None
    print(f"  {code}: N={len(y)}, N+={n_pos}, prev={y.mean():.4f}", flush=True)
    return X, y


def build_block_clean(letter, train_codes, test_codes,
                      train_users, ctx_users, query_users,
                      protein_df, disease_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}\nBlock {letter} (clean): writing {out_dir}\n{'='*70}",
          flush=True)

    # Training diseases restricted to TRAIN pool
    train_users_set = set(str(u) for u in train_users)
    train_dict = {}
    print("Training diseases (TRAIN pool):", flush=True)
    for c in train_codes:
        r = extract_disease_split(c, disease_df, protein_df, train_users_set,
                                  min_pos=30)
        if r is not None:
            train_dict[c] = r

    # Test diseases restricted to CTX pool (for in-context sampling)
    ctx_users_set = set(str(u) for u in ctx_users)
    ctx_dict = {}
    print("\nTest diseases (CTX pool):", flush=True)
    for c in test_codes:
        r = extract_disease_split(c, disease_df, protein_df, ctx_users_set,
                                  min_pos=1)
        if r is not None:
            ctx_dict[c] = r

    # Test diseases restricted to QUERY pool (for scoring)
    query_users_set = set(str(u) for u in query_users)
    qry_dict = {}
    print("\nTest diseases (QUERY pool):", flush=True)
    for c in test_codes:
        r = extract_disease_split(c, disease_df, protein_df, query_users_set,
                                  min_pos=1)
        if r is not None:
            qry_dict[c] = r

    with open(os.path.join(out_dir, "train_data.pkl"), "wb") as f:
        pickle.dump(train_dict, f)
    with open(os.path.join(out_dir, "context_data.pkl"), "wb") as f:
        pickle.dump(ctx_dict, f)
    with open(os.path.join(out_dir, "query_data.pkl"), "wb") as f:
        pickle.dump(qry_dict, f)

    print(f"\nWrote:\n"
          f"  {len(train_dict)} training diseases  → train_data.pkl\n"
          f"  {len(ctx_dict)} test-disease CTX pools → context_data.pkl\n"
          f"  {len(qry_dict)} test-disease QUERY pools → query_data.pkl",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--patient-split", required=True)
    parser.add_argument("--block-letter",
                        help="Process a single block (I, C, G, …). If omitted, "
                             "processes every block in --selection.")
    parser.add_argument("--all", action="store_true",
                        help="Process every block in --selection (alias for "
                             "omitting --block-letter).")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to processed_data_clean_{letter} "
                             "output dirs. E.g. --output-suffix _ntrain6 "
                             "writes to processed_data_clean_ntrain6_{letter}/.")
    args = parser.parse_args()

    with open(args.selection) as f:
        sel = yaml.safe_load(f)
    with open(args.patient_split) as f:
        split = yaml.safe_load(f)

    train_users = split["train"]
    ctx_users   = split["ctx"]
    query_users = split["query"]
    print(f"Split sizes  train={len(train_users)}  ctx={len(ctx_users)}  "
          f"query={len(query_users)}", flush=True)

    protein_df = load_protein_data(DATA_DIR)
    disease_df = pd.read_csv(BINARY_CSV, compression="gzip")

    letters = ([args.block_letter.upper()]
               if args.block_letter and not args.all
               else [L for L in sel["blocks"].keys()
                     if not sel["blocks"][L].get("skipped")])
    print(f"\nProcessing blocks: {letters}", flush=True)

    for L in letters:
        b = sel["blocks"][L]
        suffix = args.output_suffix
        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        out_dir = os.path.join(PROJECT_ROOT, f"processed_data_clean{suffix}_{L.lower()}")
        build_block_clean(L, b["train"], b["test"],
                          train_users, ctx_users, query_users,
                          protein_df, disease_df, out_dir)
    print("\nClean preprocessing complete.", flush=True)


if __name__ == "__main__":
    main()
