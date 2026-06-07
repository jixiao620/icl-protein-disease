import pickle

print("Loading data...")
# 用processed_data_with_q的数据
with open('processed_data_with_q/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('processed_data_with_q/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# 合并
all_data = {**train_data, **test_data}
print(f"Total diseases in dataset: {len(all_data)}")

# 定义我们感兴趣的blocks
target_blocks = {
    'O': 'Pregnancy, childbirth and puerperium (围产期)',
    'P': 'Perinatal conditions (新生儿期)',
    'Q': 'Congenital malformations (先天畸形)'
}

# 收集所有O, P, Q开头的疾病
candidates = []

for disease_code, (X, y) in all_data.items():
    code_upper = disease_code.upper()
    first_char = code_upper[0] if code_upper else ''
    
    if first_char in ['O', 'P', 'Q']:
        n_samples = len(y)
        n_pos = int(y.sum())
        prev = y.mean()
        
        # 判断block
        block = target_blocks.get(first_char, 'Unknown')
        
        candidates.append({
            'code': disease_code,
            'block': f"{first_char} - {block}",
            'n_samples': n_samples,
            'n_pos': n_pos,
            'prevalence': prev * 100
        })

# 按block和prevalence排序
candidates.sort(key=lambda x: (x['block'], -x['prevalence']))

print("\n" + "="*100)
print("All O, P, Q diseases in UK Biobank (sorted by block and prevalence)")
print("="*100)
print(f"{'Code':<15} {'Block':<45} {'N_samples':<12} {'N_pos':<10} {'Prevalence'}")
print("-"*100)

for d in candidates:
    print(f"{d['code']:<15} {d['block']:<45} {d['n_samples']:<12} {d['n_pos']:<10} {d['prevalence']:.4f}%")

print("="*100)

# 统计
by_block = {}
for d in candidates:
    block_name = d['block'].split(' - ')[0]
    if block_name not in by_block:
        by_block[block_name] = []
    by_block[block_name].append(d)

print(f"\nSummary:")
for block, diseases in sorted(by_block.items()):
    print(f"  {block} block: {len(diseases)} diseases")

# 找可以用来训练的（prevalence > 0.5%）
trainable = [d for d in candidates if d['prevalence'] > 0.5]
print(f"\n{'='*100}")
print(f"🎯 Potential TRAINING diseases (prevalence > 0.5%):")
print(f"{'='*100}")

if trainable:
    print(f"{'Code':<15} {'Block':<45} {'N_pos':<10} {'Prevalence'}")
    print("-"*100)
    for d in trainable:
        print(f"{d['code']:<15} {d['block']:<45} {d['n_pos']:<10} {d['prevalence']:.4f}%")
else:
    print("❌ No diseases with prevalence > 0.5%")

# 找rare的可以测试（0.04% - 0.2%）
testable = [d for d in candidates if 0.04 <= d['prevalence'] <= 0.2]
print(f"\n{'='*100}")
print(f"🎯 Potential TEST diseases (0.04% ≤ prevalence ≤ 0.2%):")
print(f"{'='*100}")

if testable:
    print(f"{'Code':<15} {'Block':<45} {'N_pos':<10} {'Prevalence'}")
    print("-"*100)
    for d in testable:
        print(f"{d['code']:<15} {d['block']:<45} {d['n_pos']:<10} {d['prevalence']:.4f}%")

# 保存结果
with open('congenital_perinatal_diseases.csv', 'w') as f:
    f.write("code,block,n_samples,n_pos,prevalence_percent\n")
    for d in candidates:
        f.write(f"{d['code']},{d['block']},{d['n_samples']},{d['n_pos']},{d['prevalence']:.6f}\n")

print("\n✅ Saved to congenital_perinatal_diseases.csv")

# 额外：按prevalence分组看分布
print(f"\n{'='*100}")
print("Distribution by prevalence:")
print(f"{'='*100}")
ranges = [
    ('>1%', lambda x: x > 1.0),
    ('0.5-1%', lambda x: 0.5 <= x <= 1.0),
    ('0.2-0.5%', lambda x: 0.2 <= x < 0.5),
    ('0.1-0.2%', lambda x: 0.1 <= x < 0.2),
    ('0.04-0.1%', lambda x: 0.04 <= x < 0.1),
    ('<0.04%', lambda x: x < 0.04)
]

for range_name, condition in ranges:
    count = len([d for d in candidates if condition(d['prevalence'])])
    print(f"  {range_name}: {count} diseases")
