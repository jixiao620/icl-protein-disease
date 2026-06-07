"""Create half-data version of I block processed data"""
import pickle
import numpy as np
import os

np.random.seed(42)

src_dir = '/work/jl1401/icl_protein_disease/processed_data'
dst_dir = '/work/jl1401/icl_protein_disease/processed_data_i_half'
os.makedirs(dst_dir, exist_ok=True)

# Subsample train_data to 50%
with open(f'{src_dir}/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

half_train = {}
for disease, (X, y) in train_data.items():
    n = len(y)
    half_n = n // 2
    idx = np.random.choice(n, half_n, replace=False)
    idx.sort()
    half_train[disease] = (X[idx], y[idx])
    print(f'{disease}: {n} -> {half_n} (pos={int(y[idx].sum())}, neg={int((y[idx]==0).sum())})')

with open(f'{dst_dir}/train_data.pkl', 'wb') as f:
    pickle.dump(half_train, f)

# Copy test_data unchanged
with open(f'{src_dir}/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open(f'{dst_dir}/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

# Copy feature selector and disease mapping
import shutil
shutil.copy(f'{src_dir}/feature_selector.pkl', dst_dir)
shutil.copy(f'{src_dir}/disease_mapping.pkl', dst_dir)

print('Done. Half data saved to', dst_dir)
