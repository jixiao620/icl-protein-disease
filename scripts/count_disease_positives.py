#!/usr/bin/env python3
"""
Count positive samples per disease for C and G blocks in binary_csv.gz.
Only reports diseases with >= 50 positives, excluding known test and train diseases.
"""

import gzip
import pandas as pd
import sys

DATA_PATH = "/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original/data/binary_csv.gz"

# Diseases to exclude
TEST_DISEASES = {"C34", "C18", "C43", "C53", "C54", "G40", "G20", "G30", "G35"}
TRAIN_DISEASES = {"C44", "C809", "C50", "C61", "G439", "G43", "G56", "G560"}
EXCLUDE = TEST_DISEASES | TRAIN_DISEASES

MIN_POSITIVES = 50


def extract_disease_code(col_name):
    """Extract disease code from column like 'Union#C44#C44 Other...'"""
    parts = col_name.split("#")
    if len(parts) >= 2:
        return parts[1]
    return None


def main():
    print(f"Reading column names from {DATA_PATH} ...", flush=True)

    # Read only the header first to identify relevant columns
    with gzip.open(DATA_PATH, "rt") as f:
        header_line = f.readline().rstrip("\n")

    all_cols = header_line.split(",")
    print(f"Total columns in file: {len(all_cols)}", flush=True)

    # Identify C and G block columns
    target_cols = []
    col_info = {}  # col_name -> disease_code
    for col in all_cols:
        if col.startswith("Union#C") or col.startswith("Union#G"):
            code = extract_disease_code(col)
            if code is not None and code not in EXCLUDE:
                target_cols.append(col)
                col_info[col] = code

    print(f"C/G block columns (after exclusion): {len(target_cols)}", flush=True)

    if not target_cols:
        print("No target columns found. Exiting.")
        sys.exit(1)

    # Read only the target columns in chunks to avoid OOM
    print("Reading data in chunks ...", flush=True)
    sums = pd.Series(0, index=target_cols, dtype="int64")

    chunk_size = 10_000
    reader = pd.read_csv(
        DATA_PATH,
        compression="gzip",
        usecols=target_cols,
        dtype="int8",
        chunksize=chunk_size,
    )

    n_rows = 0
    for i, chunk in enumerate(reader):
        sums = sums.add(chunk.sum(axis=0), fill_value=0)
        n_rows += len(chunk)
        if (i + 1) % 20 == 0:
            print(f"  Processed {n_rows:,} rows ...", flush=True)

    print(f"Total rows: {n_rows:,}", flush=True)

    # Build result dataframe
    result = pd.DataFrame({
        "col": target_cols,
        "disease_code": [col_info[c] for c in target_cols],
        "n_positives": [int(sums[c]) for c in target_cols],
    })

    # Filter by min positives
    result = result[result["n_positives"] >= MIN_POSITIVES].copy()

    # Split into C and G blocks
    result["block"] = result["disease_code"].apply(
        lambda x: "C" if x.startswith("C") else "G"
    )

    c_result = result[result["block"] == "C"].sort_values("n_positives", ascending=False)
    g_result = result[result["block"] == "G"].sort_values("n_positives", ascending=False)

    print("\n=== C Block Diseases (n_positives >= 50, excluding test/train) ===")
    print("disease_code\tn_positives")
    for _, row in c_result.iterrows():
        print(f"{row['disease_code']}\t{row['n_positives']}")

    print(f"\n=== G Block Diseases (n_positives >= 50, excluding test/train) ===")
    print("disease_code\tn_positives")
    for _, row in g_result.iterrows():
        print(f"{row['disease_code']}\t{row['n_positives']}")

    print(f"\nSummary: {len(c_result)} C diseases, {len(g_result)} G diseases passed the threshold.")


if __name__ == "__main__":
    main()
