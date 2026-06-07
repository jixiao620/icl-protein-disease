import pickle

print("Loading data...")
with open('processed_data_with_q/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('processed_data_with_q/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

all_data = {**train_data, **test_data}

print(f"\nTotal diseases in dataset: {len(all_data)}")

# 找所有Q开头的（不只是Q210这种细分的）
q_related = []
for disease, (X, y) in all_data.items():
    code = disease.upper()
    # Q20-Q28 (心脏), Q60-Q64 (肾脏), 或任何Q开头
    if code.startswith('Q'):
        n_pos = int(y.sum())
        prev = y.mean() * 100
        
        # 判断所属类别
        if code.startswith('Q2'):
            category = 'Q20-Q28 (Circulatory)'
        elif code.startswith('Q6'):
            category = 'Q60-Q64 (Urinary)'
        else:
            category = 'Other Q'
        
        q_related.append({
            'code': disease,
            'category': category,
            'n_samples': len(y),
            'n_pos': n_pos,
            'prevalence': prev
        })

# 按prevalence排序
q_related.sort(key=lambda x: x['prevalence'], reverse=True)

print("\n" + "="*90)
print("All Q-related diseases in UK Biobank:")
print("="*90)
print(f"{'Code':<15} {'Category':<25} {'N_samples':<12} {'N_pos':<10} {'Prevalence'}")
print("-"*90)

for d in q_related:
    print(f"{d['code']:<15} {d['category']:<25} {d['n_samples']:<12} {d['n_pos']:<10} {d['prevalence']:.4f}%")

print("="*90)
print(f"\nTotal Q-related diseases: {len(q_related)}")

# 统计
circulatory = [d for d in q_related if 'Circulatory' in d['category']]
urinary = [d for d in q_related if 'Urinary' in d['category']]

print(f"\nQ20-Q28 (Circulatory): {len(circulatory)} diseases")
print(f"Q60-Q64 (Urinary): {len(urinary)} diseases")

# 找prevalence > 0.1%的
high_prev = [d for d in q_related if d['prevalence'] > 0.1]
print(f"\nDiseases with prevalence > 0.1%: {len(high_prev)}")
if high_prev:
    print("\nPotential training candidates:")
    for d in high_prev:
        print(f"  {d['code']}: {d['n_pos']} positives ({d['prevalence']:.4f}%)")

# 保存
with open('q_related_diseases.csv', 'w') as f:
    f.write("code,category,n_samples,n_pos,prevalence_percent\n")
    for d in q_related:
        f.write(f"{d['code']},{d['category']},{d['n_samples']},{d['n_pos']},{d['prevalence']:.6f}\n")

print("\n✅ Saved to q_related_diseases.csv")
