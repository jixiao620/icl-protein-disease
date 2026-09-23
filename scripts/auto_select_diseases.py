"""
Automatically select training and held-out (test) disease codes per ICD-10
block, given a new proteomics + disease-label dataset.

Rule (locked with advisor 2026-07-28):
  Fixed scope = 3 ICD-10 chapters: I (circulatory), C (neoplasms), G (nervous).
  Within each block:
    1. Enumerate every disease code starting with the block letter.
    2. Filter to codes with >= min_positives positive cases.
    3. Rank surviving codes by PREVALENCE = positives / labeled-cohort-size,
       descending (most common first).
    4. Top n_train diseases  -> training set for this block.
    5. Bottom n_test diseases (ascending prevalence, i.e. rarest that still
       cleared the min_positives floor) -> held-out test set.
    6. Middle-prevalence diseases are unused (kept out to avoid ambiguous
       "moderately common" cases falling on either side).

Why prevalence (ratio), not raw positive count, for ranking:
  Positive count is confounded by per-disease label-recording rates. Prevalence
  (positives / cohort with a recorded label for that code) is the more honest
  proxy for "how common is this disease among people whose status is known".
  We still filter on absolute positives (>= 30) to ensure any selected code
  has enough signal to learn from.

Output:
  A YAML file (default: selection.yaml at the project root) enumerating the
  chosen train and test codes per block, plus the prevalence and positive
  count of each. This file is consumed by auto_pipeline.py.

Usage:
  python scripts/auto_select_diseases.py \\
      --data-dir data/data/Original_extracted/Original/data \\
      --output selection.yaml \\
      [--min-positives 30] [--n-train 4] [--n-test 6] \\
      [--blocks I C G]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


BLOCKS_DEFAULT = ["I", "C", "G"]


def load_disease_labels(binary_csv_path):
    """Load the disease label table."""
    if not os.path.exists(binary_csv_path):
        raise FileNotFoundError(f"Disease label file not found: {binary_csv_path}")
    print(f"Loading disease labels from {binary_csv_path} ...", flush=True)
    df = pd.read_csv(binary_csv_path, compression="gzip")
    if "userID" not in df.columns:
        raise RuntimeError(
            f"Expected a 'userID' column in {binary_csv_path} "
            "(this preprocessor assumes UKB-PPP style labels)."
        )
    print(f"  loaded: {df.shape[0]} rows x {df.shape[1]} columns", flush=True)
    return df


def enumerate_block_codes(disease_df, block_letter):
    """Return every unique ICD-10 code beginning with `block_letter`
    that appears as a `Union#{code}#...` column in disease_df.

    Filters out non-ICD-10 aggregate columns like 'Chapter XI' that also
    start with a letter matching a block (e.g. 'C', 'I') but are not real
    disease codes. Genuine ICD-10 codes always have the shape
    <letter><digit>... so we require the second character to be a digit."""
    prefix = f"Union#{block_letter}"
    codes = set()
    for col in disease_df.columns:
        if not col.startswith(prefix):
            continue
        parts = col.split("#")
        if len(parts) < 2:
            continue
        code = parts[1]
        # Real ICD-10: letter (already matched) + digit; reject spaces.
        if (code and code[0] == block_letter
                and len(code) >= 2 and code[1].isdigit()
                and " " not in code):
            codes.add(code)
    return sorted(codes)


def compute_disease_stats(disease_df, codes):
    """For each ICD-10 code, compute (labeled_N, positives, prevalence).
    labeled_N = number of participants with a non-null label for this code.
    positives = number of those with label == 1.
    prevalence = positives / labeled_N.
    Returns a DataFrame indexed by code."""
    rows = []
    for code in codes:
        col = next(
            (c for c in disease_df.columns
             if c.startswith(f"Union#{code}#") and c.split("#")[1] == code),
            None,
        )
        if col is None:
            continue
        s = disease_df[col].dropna()
        n = int(len(s))
        pos = int((s > 0.5).sum())
        prev = pos / n if n > 0 else 0.0
        rows.append({"code": code, "labeled_n": n,
                     "positives": pos, "prevalence": prev})
    if not rows:
        return pd.DataFrame(columns=["code", "labeled_n",
                                     "positives", "prevalence"])
    return pd.DataFrame(rows).set_index("code")


def select_train_test(stats, min_positives, n_train, n_test):
    """Apply the selection rule to a per-block stats DataFrame.
    Returns (train_codes, test_codes, dropped_reason)."""
    # 1. min-positives filter
    survivors = stats[stats["positives"] >= min_positives].copy()
    if len(survivors) == 0:
        return [], [], (f"no codes with >= {min_positives} positives "
                        "(block skipped)")
    if len(survivors) < n_train + n_test:
        return [], [], (
            f"only {len(survivors)} codes clear the >= {min_positives} "
            f"positives filter; need at least n_train + n_test = "
            f"{n_train + n_test} (block skipped)"
        )

    # 2. Rank by prevalence, descending
    survivors_sorted = survivors.sort_values("prevalence", ascending=False)

    # 3. Top n_train -> training
    train = survivors_sorted.head(n_train).index.tolist()

    # 4. Bottom n_test (by ascending prevalence) -> test
    test = survivors_sorted.tail(n_test).index.tolist()

    # 5. Middle codes discarded (implicit)
    return train, test, None


def build_selection(disease_df, blocks, min_positives, n_train, n_test):
    """Produce the full selection payload as a plain dict (YAML-serializable)."""
    result = {
        "meta": {
            "rule": "prevalence-based split per ICD-10 block",
            "min_positives": min_positives,
            "n_train": n_train,
            "n_test": n_test,
        },
        "blocks": {},
    }
    for block in blocks:
        print(f"\nBlock {block} — enumerating ...", flush=True)
        codes = enumerate_block_codes(disease_df, block)
        print(f"  found {len(codes)} unique codes in the label file",
              flush=True)
        stats = compute_disease_stats(disease_df, codes)
        train, test, skip_reason = select_train_test(
            stats, min_positives, n_train, n_test,
        )

        block_out = {}
        if skip_reason is not None:
            print(f"  ⚠ SKIPPED: {skip_reason}", flush=True)
            block_out["skipped"] = True
            block_out["skip_reason"] = skip_reason
        else:
            print(f"  train ({len(train)}): {train}", flush=True)
            print(f"  test  ({len(test)}): {test}",  flush=True)
            block_out["train"] = train
            block_out["test"]  = test

        # Persist the per-code stats (handy for debugging / paper reporting)
        details = {}
        for code, row in stats.iterrows():
            details[code] = {
                "labeled_n":  int(row["labeled_n"]),
                "positives":  int(row["positives"]),
                "prevalence": float(round(row["prevalence"], 6)),
            }
        block_out["diseases"] = details
        result["blocks"][block] = block_out
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Auto-select ICD-10 train/test diseases per block."
    )
    parser.add_argument("--data-dir",  required=True,
                        help="Directory containing binary_csv.gz "
                             "(disease labels).")
    parser.add_argument("--output",    default="selection.yaml",
                        help="Where to write the selection YAML "
                             "(default: selection.yaml in cwd).")
    parser.add_argument("--min-positives", type=int, default=30,
                        help="Minimum absolute positive count required "
                             "for a code to be considered (default 30).")
    parser.add_argument("--n-train", type=int, default=4,
                        help="Number of training diseases per block "
                             "(default 4, top by prevalence).")
    parser.add_argument("--n-test",  type=int, default=6,
                        help="Number of held-out test diseases per block "
                             "(default 6, bottom-prevalence survivors).")
    parser.add_argument("--blocks",  nargs="+", default=BLOCKS_DEFAULT,
                        help="Block letters to process (default: I C G).")
    args = parser.parse_args()

    binary_csv = os.path.join(args.data_dir, "binary_csv.gz")
    disease_df = load_disease_labels(binary_csv)

    selection = build_selection(
        disease_df, args.blocks,
        min_positives=args.min_positives,
        n_train=args.n_train, n_test=args.n_test,
    )

    with open(args.output, "w") as f:
        yaml.safe_dump(selection, f, sort_keys=False, default_flow_style=False)

    print(f"\nSelection written to {args.output}", flush=True)
    print("Summary (train / test counts per block):", flush=True)
    for block, out in selection["blocks"].items():
        if out.get("skipped"):
            print(f"  {block}: SKIPPED — {out['skip_reason']}", flush=True)
        else:
            print(f"  {block}: {len(out['train'])} train / "
                  f"{len(out['test'])} test", flush=True)


if __name__ == "__main__":
    main()
