"""Count positive cases per disease from binary_csv.gz — submit via sbatch, not login node."""
import gzip
import csv
import sys
from collections import defaultdict

DATA = '/work/jl1401/icl_protein_disease/data/data/Original_extracted/Original/data/binary_csv.gz'

counts = defaultdict(int)
total_patients = 0

with gzip.open(DATA, 'rt') as f:
    reader = csv.reader(f)
    header = next(reader)
    # Extract I-block columns (code starts with #I, not #Block)
    i_cols = [(i, c) for i, c in enumerate(header)
              if '#I' in c and '#Block' not in c and i > 0]

    for row in reader:
        total_patients += 1
        for idx, col in i_cols:
            if idx < len(row) and row[idx] == '1':
                counts[col] += 1

print(f"Total patients: {total_patients}\n")
print(f"{'Code':<12} {'Count':>7} {'Prevalence':>12}  Description")
print("-" * 70)

# Parse code and sort by count descending
parsed = []
for col, cnt in counts.items():
    # col format: Union#CODE#CODE Description
    parts = col.split('#')
    code = parts[1] if len(parts) > 1 else col
    desc = parts[2] if len(parts) > 2 else ''
    parsed.append((code, cnt, desc))

parsed.sort(key=lambda x: -x[1])
for code, cnt, desc in parsed:
    prev = cnt / total_patients * 100
    print(f"{code:<12} {cnt:>7} {prev:>11.2f}%  {desc[:50]}")
