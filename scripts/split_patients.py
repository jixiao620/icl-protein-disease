"""
Patient-disjoint 3-way split (train / context / query) for the CLEAN pipeline.

Motivation:
  The AUTOMATED pipeline (auto_pipeline.py + auto_eval.py) trains and evaluates
  on the same patient cohort — every patient's plasma is seen at training time
  with training-disease labels, and the same patients can appear at inference
  time as either context or query for a test disease. Reviewer concern #1
  called this out as a leakage source: the model can memorise a patient's
  protein signature and exploit comorbidity correlations at eval.

  This CLEAN pipeline eliminates that leakage by partitioning userIDs
  disjointly. The training pool is used only to fit the ICL model on
  training diseases; the context pool is used only to sample in-context
  examples at eval; the query pool is used only for scoring test-disease
  AUROC/AUPRC/Brier/ECE.

Split rule (locked with advisor 2026-08-04):
  * Fraction: train 60% / context 20% / query 20%
  * Stratified by: union label = "positive for at least one test disease
    across all requested blocks". This keeps test-disease positives (the
    scarce resource — usually ~30 per disease) proportional across pools.
  * Random state: 42 (deterministic — the same raw data always produces
    the same patient_split.yaml).

Usage:
  python scripts/split_patients.py \\
      --data-dir data/data/Original_extracted/Original/data \\
      --selection selection.yaml \\
      --output patient_split.yaml \\
      [--train-frac 0.6 --ctx-frac 0.2 --query-frac 0.2] [--seed 42]

Output file (YAML):
  meta:
    train_frac: 0.6
    ctx_frac:   0.2
    query_frac: 0.2
    seed:       42
    stratify:   union-of-test-disease-positives
  train: [userID1, userID2, ...]
  ctx:   [...]
  query: [...]
  audit:
    per_disease:
      I456: {train: 18, ctx: 6, query: 6}
      C676: ...
      ...

Downstream consumers:
  preprocess_clean.py         reads this file and produces
                              processed_data_clean_{letter}/ with
                              train_data.pkl / context_data.pkl / query_data.pkl
  eval_clean.py               reads context_data + query_data pickles
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def find_disease_column(disease_df, code):
    return next(
        (c for c in disease_df.columns
         if c.startswith(f"Union#{code}#") and c.split("#")[1] == code),
        None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing binary_csv.gz")
    parser.add_argument("--selection", required=True,
                        help="selection.yaml (from auto_select_diseases.py); "
                             "used to identify test diseases for stratification")
    parser.add_argument("--output", default="patient_split.yaml")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--ctx-frac",   type=float, default=0.20)
    parser.add_argument("--query-frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tot = args.train_frac + args.ctx_frac + args.query_frac
    if abs(tot - 1.0) > 1e-6:
        parser.error(f"train + ctx + query = {tot:.4f}, must sum to 1.0")

    # ---- Load labels ----
    binary_csv = os.path.join(args.data_dir, "binary_csv.gz")
    print(f"Loading labels from {binary_csv} ...", flush=True)
    disease_df = pd.read_csv(binary_csv, compression="gzip")
    print(f"  {disease_df.shape[0]} rows × {disease_df.shape[1]} cols", flush=True)
    if "userID" not in disease_df.columns:
        raise RuntimeError("binary_csv.gz must contain a 'userID' column")

    # ---- Read test diseases from selection ----
    with open(args.selection) as f:
        sel = yaml.safe_load(f)
    test_codes = []
    for letter, out in sel.get("blocks", {}).items():
        if out.get("skipped"):
            continue
        for c in out.get("test", []):
            test_codes.append(c)
    print(f"\nTest diseases from selection ({len(test_codes)}): {test_codes}",
          flush=True)

    # ---- Build union-of-test-positive label per user ----
    all_users = disease_df["userID"].astype(str).values
    union_pos = np.zeros(len(disease_df), dtype=np.int8)
    for code in test_codes:
        col = find_disease_column(disease_df, code)
        if col is None:
            print(f"  {code}: column not found — skipping stratifier",
                  flush=True); continue
        vals = disease_df[col].fillna(0).values
        union_pos |= (vals > 0.5).astype(np.int8)

    n_union_pos = int(union_pos.sum())
    print(f"Users positive for ≥1 test disease: {n_union_pos} "
          f"({n_union_pos/len(union_pos)*100:.3f}%)", flush=True)

    # ---- Stratified 3-way split ----
    # Sklearn only does 2-way splits, so we split twice.
    tv_frac = args.ctx_frac + args.query_frac
    rest_frac = args.ctx_frac / tv_frac        # ctx portion of the (ctx+query) leftover

    train_idx, rest_idx = train_test_split(
        np.arange(len(all_users)),
        test_size=tv_frac,
        stratify=union_pos,
        random_state=args.seed,
    )
    ctx_idx, query_idx = train_test_split(
        rest_idx,
        test_size=1.0 - rest_frac,
        stratify=union_pos[rest_idx],
        random_state=args.seed,
    )
    train_users = sorted(all_users[train_idx].tolist())
    ctx_users   = sorted(all_users[ctx_idx].tolist())
    query_users = sorted(all_users[query_idx].tolist())
    print(f"\nSplit sizes  train={len(train_users):>7d}  "
          f"ctx={len(ctx_users):>7d}  query={len(query_users):>7d}",
          flush=True)

    # ---- Audit: per-disease positive counts in each pool ----
    train_set, ctx_set, query_set = map(set, (train_users, ctx_users, query_users))
    audit = {}
    for code in test_codes:
        col = find_disease_column(disease_df, code)
        if col is None: continue
        sub = disease_df[["userID", col]].dropna()
        # positives only
        pos_users = sub.loc[sub[col] > 0.5, "userID"].astype(str).values
        row = {"pos_train": int(sum(u in train_set for u in pos_users)),
               "pos_ctx":   int(sum(u in ctx_set   for u in pos_users)),
               "pos_query": int(sum(u in query_set for u in pos_users))}
        row["pos_total"] = row["pos_train"] + row["pos_ctx"] + row["pos_query"]
        audit[code] = row

    # ---- Write output ----
    payload = {
        "meta": {
            "train_frac": args.train_frac,
            "ctx_frac":   args.ctx_frac,
            "query_frac": args.query_frac,
            "seed":       args.seed,
            "stratify":   "union-of-test-disease-positives",
            "n_users_total": int(len(all_users)),
            "n_users_train": len(train_users),
            "n_users_ctx":   len(ctx_users),
            "n_users_query": len(query_users),
        },
        "train": train_users,
        "ctx":   ctx_users,
        "query": query_users,
        "audit": {"per_disease": audit},
    }
    with open(args.output, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    # ---- Console summary ----
    print("\nPer-test-disease positive counts in each pool:", flush=True)
    print(f"  {'code':<8} {'train':>6} {'ctx':>6} {'query':>6} {'total':>6}",
          flush=True)
    for code, row in audit.items():
        print(f"  {code:<8} {row['pos_train']:>6} {row['pos_ctx']:>6} "
              f"{row['pos_query']:>6} {row['pos_total']:>6}", flush=True)
    n_low = sum(1 for r in audit.values()
                if min(r["pos_ctx"], r["pos_query"]) < 3)
    if n_low:
        print(f"\n  ⚠ {n_low} test disease(s) have < 3 positives in "
              f"ctx or query — eval variance will be high", flush=True)
    print(f"\nSplit written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
