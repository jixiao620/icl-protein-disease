"""
Preprocess additional training diseases for C and G blocks from raw data.
Outputs: processed_data_c_v6/train_data.pkl, processed_data_g_v6/train_data.pkl
         (also copies existing test_data.pkl from processed_data_c/g)
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = "/work/jl1401/icl_protein_disease"
DATA_DIR = os.path.join(PROJECT_ROOT, "data/data/Original_extracted/Original/data")
BINARY_CSV = os.path.join(DATA_DIR, "binary_csv.gz")

# C block: existing train + new additions (never include test: C34,C18,C43,C53,C54)
C_TRAIN_EXISTING = ["C44", "C809", "C50", "C61"]
C_TRAIN_NEW = [
    "C16",   # stomach cancer
    "C22",   # liver cancer
    "C25",   # pancreatic cancer
    "C56",   # ovarian cancer
    "C67",   # bladder cancer
    "C71",   # brain tumor/glioma
    "C73",   # thyroid cancer
    "C85",   # non-Hodgkin lymphoma
    "C90",   # multiple myeloma
    "C92",   # myeloid leukemia
]
C_TEST = ["C34", "C18", "C43", "C53", "C54"]

# G block: existing train + new additions (never include test: G40,G20,G30,G35)
G_TRAIN_EXISTING = ["G439", "G43", "G56", "G560"]
G_TRAIN_NEW = [
    "G10",   # Huntington's disease (neurodegenerative)
    "G12",   # spinal muscular atrophy
    "G41",   # status epilepticus (seizure-related, like G40)
    "G45",   # TIA / vascular neurological
    "G47",   # sleep disorders (common, good signal)
    "G61",   # Guillain-Barre (autoimmune, like G35 MS)
    "G71",   # myopathy/muscular dystrophy
    "G80",   # cerebral palsy
    "G93",   # other brain disorders
    "G31",   # other degenerative brain (like G30 Alzheimer's)
]
G_TEST = ["G40", "G20", "G30", "G35"]


def load_protein_data(data_dir):
    """Load all protein files (row-split across files). Returns df with userID + protein cols.
    xaa.gz has header row; all other files have no header (same columns, different patients)."""
    protein_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith('xa') and f.endswith('.gz')
    ])
    print(f"\nLoading {len(protein_files)} protein files (row-split)...")
    dfs = []
    col_names = None
    for i, fname in enumerate(protein_files):
        fpath = os.path.join(data_dir, fname)
        print(f"  [{i+1}/{len(protein_files)}] {fname}", flush=True)
        if col_names is None:
            # First file: read with header to get column names
            df = pd.read_csv(fpath, compression='gzip')
            col_names = df.columns.tolist()
        else:
            # Remaining files: no header, use same column names
            df = pd.read_csv(fpath, compression='gzip', header=None, names=col_names)
        dfs.append(df)
    protein_df = pd.concat(dfs, ignore_index=True)
    print(f"Protein data: {protein_df.shape[0]} patients x {protein_df.shape[1]-1} proteins")
    return protein_df


def load_disease_labels(binary_csv):
    """Load binary disease labels. Returns df with userID + disease columns."""
    print(f"\nLoading disease labels from {binary_csv}...")
    df = pd.read_csv(binary_csv, compression='gzip')
    print(f"Disease data: {df.shape[0]} patients x {df.shape[1]} columns")
    return df


def find_disease_column(disease_df, disease_code):
    """Find the full column name for a disease code like 'C44'."""
    target = f"Union#{disease_code}#"
    matches = [c for c in disease_df.columns if c.startswith(target)]
    if not matches:
        return None
    if len(matches) > 1:
        # prefer exact match (e.g., "C44" not "C440")
        exact = [c for c in matches if c.split('#')[1] == disease_code]
        if exact:
            return exact[0]
    return matches[0]


def extract_disease(disease_code, disease_df, protein_df, min_positives=30):
    """Extract (X, y) for one disease. Returns None if insufficient positives."""
    col = find_disease_column(disease_df, disease_code)
    if col is None:
        print(f"  {disease_code}: column not found, skipping")
        return None

    # Merge disease label with protein data
    label_df = disease_df[['userID', col]].dropna()
    merged = protein_df.merge(label_df, on='userID', how='inner')

    y = merged[col].values.astype(np.float32)
    X = merged.drop(columns=['userID', col]).values.astype(np.float32)

    n_pos = int(y.sum())
    n_total = len(y)
    prevalence = y.mean()

    if n_pos < min_positives:
        print(f"  {disease_code}: {n_pos} positives (too few, need {min_positives}), skipping")
        return None

    print(f"  {disease_code}: {n_total} patients, {n_pos} positives ({prevalence:.4f})")
    return X, y


def process_block(block, train_existing, train_new, test_codes,
                  disease_df, protein_df, output_dir, existing_processed_dir):
    """Process one disease block, save expanded train pkl + copy test pkl."""
    os.makedirs(output_dir, exist_ok=True)
    all_train = train_existing + train_new
    train_data = {}

    print(f"\n{'='*60}")
    print(f"Processing {block} block: {len(all_train)} training diseases")
    print(f"{'='*60}")

    for disease in all_train:
        result = extract_disease(disease, disease_df, protein_df)
        if result is not None:
            train_data[disease] = result

    train_out = os.path.join(output_dir, 'train_data.pkl')
    with open(train_out, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\nSaved {len(train_data)} training diseases to {train_out}")

    # Copy existing test pkl
    src_test = os.path.join(existing_processed_dir, 'test_data.pkl')
    dst_test = os.path.join(output_dir, 'test_data.pkl')
    if os.path.exists(src_test):
        import shutil
        shutil.copy2(src_test, dst_test)
        print(f"Copied test_data.pkl from {existing_processed_dir}")
    else:
        print(f"WARNING: no test_data.pkl found at {src_test}")

    # Also copy disease_mapping if exists
    src_map = os.path.join(existing_processed_dir, 'disease_mapping.pkl')
    if os.path.exists(src_map):
        import shutil
        shutil.copy2(src_map, os.path.join(output_dir, 'disease_mapping.pkl'))

    print(f"\n{block} block summary:")
    for code, (X, y) in train_data.items():
        tag = "(existing)" if code in train_existing else "(NEW)"
        print(f"  {code} {tag}: {len(y)} patients, {int(y.sum())} positives ({y.mean():.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['c', 'g', 'both'], default='both')
    parser.add_argument('--min_positives', type=int, default=30)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Preprocessing expanded training diseases")
    print(f"{'='*60}")

    # Load raw data once
    protein_df = load_protein_data(DATA_DIR)
    disease_df = load_disease_labels(BINARY_CSV)

    if args.block in ('c', 'both'):
        process_block(
            block='C',
            train_existing=C_TRAIN_EXISTING,
            train_new=C_TRAIN_NEW,
            test_codes=C_TEST,
            disease_df=disease_df,
            protein_df=protein_df,
            output_dir=os.path.join(PROJECT_ROOT, 'processed_data_c_v6'),
            existing_processed_dir=os.path.join(PROJECT_ROOT, 'processed_data_c'),
        )

    if args.block in ('g', 'both'):
        process_block(
            block='G',
            train_existing=G_TRAIN_EXISTING,
            train_new=G_TRAIN_NEW,
            test_codes=G_TEST,
            disease_df=disease_df,
            protein_df=protein_df,
            output_dir=os.path.join(PROJECT_ROOT, 'processed_data_g_v6'),
            existing_processed_dir=os.path.join(PROJECT_ROOT, 'processed_data_g'),
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
