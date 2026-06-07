#!/usr/bin/env python -u
import sys
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

import pickle
import gzip
import csv
import numpy as np
from pathlib import Path

print("="*60, flush=True)
print("📊 Processing O Diseases Data", flush=True)
print("="*60, flush=True)

# Load protein data from split gz files (same as I diseases processing)
print("🧬 Loading protein data from split files...", flush=True)

protein_files = [
    'data/data/data/xaa.gz',
    'data/data/data/xaa_2.gz',
    'data/data/data/xab.gz',
    'data/data/data/xac.gz',
    'data/data/data/xad.gz',
    'data/data/data/xae.gz',
    'data/data/data/xaf.gz',
    'data/data/data/xag.gz',
    'data/data/data/xah.gz',
    'data/data/data/xai.gz',
    'data/data/data/xaj.gz'
]

print(f"Found {len(protein_files)} protein files", flush=True)

protein_data = []
protein_header = None

for i, file_path in enumerate(protein_files):
    print(f"  Reading {Path(file_path).name} ({i+1}/{len(protein_files)})...", flush=True)
    
    with gzip.open(file_path, 'rt') as f:
        reader = csv.reader(f)
        
        if i == 0:
            # First file has header
            protein_header = next(reader)
            print(f"    ✅ Header detected: {len(protein_header)} columns", flush=True)
        
        for row in reader:
            # Skip eid (first column), keep protein values
            protein_row = [float(x) if x else 0.0 for x in row[1:]]
            protein_data.append(protein_row)

protein_features = np.array(protein_data)
n_patients = len(protein_features)
n_proteins = protein_features.shape[1]

print(f"✅ Total protein data: {n_patients} patients × {n_proteins} features", flush=True)

if n_proteins == 0:
    print("❌ ERROR: No protein features loaded!", flush=True)
    sys.exit(1)

# Load binary disease data
print("\n📂 Loading disease labels from binary_csv.gz...", flush=True)
with gzip.open('data/data/data/binary_csv.gz', 'rt') as f:
    reader = csv.DictReader(f)
    
    # Get column names from first row
    first_row = next(reader)
    all_columns = list(first_row.keys())
    
    print(f"✅ Loaded {len(all_columns)} columns", flush=True)
    
    # Find O disease columns: Union#CODE#Description
    disease_info = {}
    for col in all_columns:
        if col.startswith('Union#'):
            parts = col.split('#')
            if len(parts) >= 3:
                code = parts[1]
                if code.startswith('O'):  # Only O diseases
                    description = parts[2]
                    disease_info[col] = {'code': code, 'desc': description}
    
    print(f"Found {len(disease_info)} O disease columns", flush=True)
    
    # Initialize data storage
    disease_labels = {col: [] for col in disease_info.keys()}
    
    # Read first row
    for col in disease_info.keys():
        val = first_row[col]
        disease_labels[col].append(int(val) if val != '' else 0)
    
    # Read remaining rows
    sample_count = 1
    for i, row in enumerate(reader):
        sample_count += 1
        for col in disease_info.keys():
            val = row[col]
            disease_labels[col].append(int(val) if val != '' else 0)
        
        if sample_count % 10000 == 0:
            print(f"  Processed {sample_count} samples...", flush=True)

print(f"Total disease label rows: {sample_count}", flush=True)

# Organize by disease code
print("\nOrganizing diseases...", flush=True)
o_diseases = {}

for col, info in disease_info.items():
    code = info['code']
    y = np.array(disease_labels[col])
    n_pos = int(y.sum())
    prev = (n_pos / len(y)) * 100
    
    o_diseases[code] = {
        'y': y,
        'n_pos': n_pos,
        'prevalence': prev,
        'description': info['desc']
    }

# Manually select training and testing diseases
rare_threshold = 0.05  # 1/2000

# Training: O80, O70 (highest prevalence)
train_diseases = ['O80', 'O70']

# Testing: rare diseases (prevalence < 0.05%, n_pos >= 20)
test_diseases = []

print(f"\n{'='*60}", flush=True)
print("O Diseases Classification:", flush=True)
print(f"{'='*60}", flush=True)

print("\n🔹 TRAINING Diseases (O80, O70):", flush=True)
for code in train_diseases:
    if code in o_diseases:
        data = o_diseases[code]
        print(f"  {code}: {data['n_pos']} ({data['prevalence']:.3f}%)", flush=True)
    else:
        print(f"  ⚠️  {code} not found in data!", flush=True)

print(f"\n🔸 RARE Diseases (prevalence < {rare_threshold}%):", flush=True)
for code, data in sorted(o_diseases.items(), key=lambda x: x[1]['prevalence'], reverse=True):
    if data['prevalence'] < rare_threshold and data['n_pos'] >= 20:
        print(f"  {code}: {data['n_pos']} ({data['prevalence']:.4f}%)", flush=True)
        test_diseases.append(code)

print(f"\nTotal training diseases: {len(train_diseases)}", flush=True)
print(f"Total testing (rare) diseases: {len(test_diseases)}", flush=True)

# Match protein data with disease labels
print(f"\n🔗 Matching protein data with disease labels...", flush=True)
print(f"Protein patients: {n_patients}", flush=True)
print(f"Disease label samples: {sample_count}", flush=True)

if n_patients != sample_count:
    print(f"⚠️  Sample size mismatch!", flush=True)
    print(f"Will truncate to minimum length", flush=True)
    min_len = min(n_patients, sample_count)
else:
    print(f"✓ Sample sizes match!", flush=True)
    min_len = n_patients

# Create train/test data structure
Path('processed_data_o').mkdir(exist_ok=True)

train_data = {}
test_data = {}

print(f"\n📦 Creating training data...", flush=True)
for code in train_diseases:
    if code in o_diseases:
        X = protein_features[:min_len]
        y = o_diseases[code]['y'][:min_len]
        
        train_data[code] = (X, y)
        print(f"  ✅ {code}: X {X.shape}, y {y.shape}, {int(y.sum())} positives", flush=True)

print(f"\n📦 Creating testing data...", flush=True)
for code in test_diseases:
    X = protein_features[:min_len]
    y = o_diseases[code]['y'][:min_len]
    
    test_data[code] = (X, y)
    print(f"  ✅ {code}: X {X.shape}, y {y.shape}, {int(y.sum())} positives", flush=True)

# Save
with open('processed_data_o/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('processed_data_o/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print(f"\n{'='*60}", flush=True)
print("✅ Processing Complete!", flush=True)
print(f"{'='*60}", flush=True)
print(f"Saved to processed_data_o/", flush=True)
print(f"  train_data.pkl: {len(train_data)} diseases (O80, O70)", flush=True)
print(f"  test_data.pkl: {len(test_diseases)} rare diseases", flush=True)

