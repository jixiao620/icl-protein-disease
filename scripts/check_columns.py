
#!/usr/bin/env python -u

import csv

import gzip



print("Checking column names in binary_csv.gz...")

with gzip.open('data/data/data/binary_csv.gz', 'rt') as f:

    reader = csv.DictReader(f)

    columns = reader.fieldnames

    

print(f"\nTotal columns: {len(columns)}")

print("\nFirst 50 columns:")

for i, col in enumerate(columns[:50]):

    print(f"  {i}: {col}")



print("\n...")

print("\nLast 50 columns:")

for i, col in enumerate(columns[-50:], start=len(columns)-50):

    print(f"  {i}: {col}")



# 找所有可能是疾病的列

print("\nLooking for O, P, Q prefixed columns...")

opq_cols = [col for col in columns if col and col[0].upper() in ['O', 'P', 'Q']]

print(f"Found {len(opq_cols)} O/P/Q columns")

if opq_cols:

    print("Examples:")

    for col in opq_cols[:20]:

        print(f"  {col}")

