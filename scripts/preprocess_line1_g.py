"""
Preprocess G block pathway training data for Line 1 pathway validation.
Creates processed_data_line1_g/train_data.pkl with all 12 pathway diseases.
Single pkl file; each training config filters to its subset.
"""

import os, pickle
import numpy as np
import pandas as pd

PROJECT  = '/work/jl1401/icl_protein_disease'
DATA_DIR = os.path.join(PROJECT, 'data/data/Original_extracted/Original/data')
OUT_DIR  = os.path.join(PROJECT, 'processed_data_line1_g')

# All diseases across 3 pathways (pathways defined in configs)
ALL_DISEASES = [
    # PN: Peripheral Nerve / Mononeuropathy
    'G54', 'G55', 'G57', 'G58',
    # ND: Neurodegenerative Brain
    'G20', 'G30', 'G31', 'G35',
    # HS: Headache / Sleep
    'G44', 'G47', 'G50', 'G51',
]


def load_protein_data():
    fnames = sorted(f for f in os.listdir(DATA_DIR) if f.startswith('xa') and f.endswith('.gz'))
    dfs, cols = [], None
    for fn in fnames:
        fp = os.path.join(DATA_DIR, fn)
        if cols is None:
            df = pd.read_csv(fp, compression='gzip')
            cols = df.columns.tolist()
        else:
            df = pd.read_csv(fp, compression='gzip', header=None, names=cols)
        dfs.append(df)
        print(f"  {fn}: {len(df)} rows", flush=True)
    protein_df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {protein_df.shape}", flush=True)
    return protein_df


def find_col(disease_df, code):
    """Find full column name for disease code in binary_csv (format: Union#CODE#Desc)."""
    matches = [c for c in disease_df.columns if c.startswith(f'Union#{code}#')]
    if not matches:
        return None
    exact = [c for c in matches if c.split('#')[1] == code]
    return exact[0] if exact else matches[0]


def main():
    print("Loading protein data...", flush=True)
    protein_df = load_protein_data()

    print("Loading binary CSV...", flush=True)
    disease_df = pd.read_csv(os.path.join(DATA_DIR, 'binary_csv.gz'), compression='gzip')
    print(f"Binary CSV: {disease_df.shape}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    train_data = {}

    for code in ALL_DISEASES:
        col = find_col(disease_df, code)
        if col is None:
            print(f"  {code}: NOT FOUND in binary_csv, skipping", flush=True)
            continue
        label_df = disease_df[['userID', col]].dropna()
        merged = protein_df.merge(label_df, on='userID', how='inner')
        y = merged[col].values.astype(np.float32)
        X = merged.drop(columns=['userID', col]).values.astype(np.float32)
        if not np.isfinite(X).all():
            X[~np.isfinite(X)] = 0.0
        train_data[code] = (X, y)
        print(f"  {code}: N={len(y)}, N+={int(y.sum())} ({y.mean():.4f})", flush=True)

    out = os.path.join(OUT_DIR, 'train_data.pkl')
    with open(out, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\nSaved {len(train_data)}/{len(ALL_DISEASES)} diseases to {out}", flush=True)

    print("\nSummary by pathway:")
    pn = {k: v for k, v in train_data.items() if k in ['G54','G55','G57','G58']}
    nd = {k: v for k, v in train_data.items() if k in ['G20','G30','G31','G35']}
    hs = {k: v for k, v in train_data.items() if k in ['G44','G47','G50','G51']}
    for label, subset in [('PN', pn), ('ND', nd), ('HS', hs)]:
        total_n = sum(len(y) for _, (_, y) in subset.items())
        total_pos = sum(int(y.sum()) for _, (_, y) in subset.items())
        print(f"  {label}: {list(subset.keys())}, total N={total_n}, N+={total_pos}", flush=True)


if __name__ == '__main__':
    main()
