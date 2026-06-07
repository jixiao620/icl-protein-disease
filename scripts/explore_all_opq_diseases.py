#!/usr/bin/env python -u
import csv
import gzip
import json

print("Loading binary disease data (CSV)...")
with gzip.open('data/data/data/binary_csv.gz', 'rt') as f:
    reader = csv.DictReader(f)
    
    # 读取第一行获取所有列名
    first_row = next(reader)
    all_columns = list(first_row.keys())
    
    print(f"Total columns: {len(all_columns)}")
    
    # 解析列名：Union#CODE#Description
    # 提取疾病code (第二个#之间的部分)
    disease_info = {}
    for col in all_columns:
        if col.startswith('Union#'):
            parts = col.split('#')
            if len(parts) >= 3:
                code = parts[1]  # 例如 A01, I10, Q210等
                description = parts[2]
                disease_info[col] = {'code': code, 'desc': description}
    
    print(f"Disease columns: {len(disease_info)}")
    
    # 初始化计数器
    disease_counts = {col: 0 for col in disease_info.keys()}
    n_samples = 1
    
    # 统计第一行
    for col in disease_info.keys():
        if first_row[col] == '1':
            disease_counts[col] += 1
    
    # 继续读取剩余行
    for row in reader:
        n_samples += 1
        for col in disease_info.keys():
            if row[col] == '1':
                disease_counts[col] += 1
        
        if n_samples % 10000 == 0:
            print(f"  Processed {n_samples} rows...", flush=True)

print(f"\nTotal samples: {n_samples}")

# 收集O, P, Q diseases
opq_diseases = []

for col, info in disease_info.items():
    code = info['code']
    first_char = code[0].upper() if code else ''
    
    if first_char in ['O', 'P', 'Q']:
        n_pos = disease_counts[col]
        prev = (n_pos / n_samples) * 100
        
        # 判断block
        if first_char == 'O':
            block = 'O (Pregnancy/childbirth)'
        elif first_char == 'P':
            block = 'P (Perinatal)'
        elif first_char == 'Q':
            block = 'Q (Congenital)'
        
        opq_diseases.append({
            'code': code,
            'name': info['desc'],
            'block': block,
            'n_samples': n_samples,
            'n_pos': n_pos,
            'prevalence': prev
        })

# 按block和prevalence排序
opq_diseases.sort(key=lambda x: (x['block'], -x['prevalence']))

print("\n" + "="*120)
print("All O, P, Q diseases in UK Biobank")
print("="*120)
print(f"{'Code':<10} {'Block':<25} {'N_samples':<12} {'N_pos':<10} {'Prev%':<10} {'Disease Name'}")
print("-"*120)

for d in opq_diseases:
    name_short = d['name'][:60] if len(d['name']) > 60 else d['name']
    print(f"{d['code']:<10} {d['block']:<25} {d['n_samples']:<12} {d['n_pos']:<10} {d['prevalence']:>8.4f}% {name_short}")

print("="*120)

# 统计
o_diseases = [d for d in opq_diseases if d['block'].startswith('O')]
p_diseases = [d for d in opq_diseases if d['block'].startswith('P')]
q_diseases = [d for d in opq_diseases if d['block'].startswith('Q')]

print(f"\nSummary by block:")
print(f"  O block: {len(o_diseases)} diseases")
print(f"  P block: {len(p_diseases)} diseases")
print(f"  Q block: {len(q_diseases)} diseases")
print(f"  Total: {len(opq_diseases)} diseases")

# 找可以训练的（prevalence > 0.5%）
trainable = [d for d in opq_diseases if d['prevalence'] > 0.5]
print(f"\n{'='*120}")
print(f"🎯 TRAINING candidates (prevalence > 0.5%):")
print(f"{'='*120}")

if trainable:
    print(f"{'Code':<10} {'Block':<25} {'N_pos':<10} {'Prev%':<10} {'Disease Name'}")
    print("-"*120)
    for d in trainable:
        name_short = d['name'][:60]
        print(f"{d['code']:<10} {d['block']:<25} {d['n_pos']:<10} {d['prevalence']:>8.4f}% {name_short}")
else:
    print("❌ No diseases with prevalence > 0.5%")

# 找rare测试集（0.04% - 0.2%）
testable = [d for d in opq_diseases if 0.04 <= d['prevalence'] <= 0.2]
print(f"\n{'='*120}")
print(f"🎯 TEST candidates (0.04% ≤ prevalence ≤ 0.2%, rare diseases):")
print(f"{'='*120}")

if testable:
    print(f"{'Code':<10} {'Block':<25} {'N_pos':<10} {'Prev%':<10} {'Disease Name'}")
    print("-"*120)
    for d in testable:
        name_short = d['name'][:60]
        print(f"{d['code']:<10} {d['block']:<25} {d['n_pos']:<10} {d['prevalence']:>8.4f}% {name_short}")

# 保存结果
with open('all_opq_diseases.csv', 'w') as f:
    f.write("code,block,name,n_samples,n_pos,prevalence_percent\n")
    for d in opq_diseases:
        name_clean = d['name'].replace(',', ';').replace('\n', ' ')
        f.write(f"{d['code']},{d['block']},{name_clean},{d['n_samples']},{d['n_pos']},{d['prevalence']:.6f}\n")

print("\n✅ Saved to all_opq_diseases.csv")
