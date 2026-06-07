import pickle
import numpy as np

# 加载所有数据
print("Loading data...", flush=True)
with open('processed_data_with_q/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('processed_data_with_q/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# 合并
all_data = {**train_data, **test_data}
print(f"Total diseases: {len(all_data)}", flush=True)

# 找所有Q开头的
q_diseases = []
for disease, (X, y) in all_data.items():
    if disease.startswith('Q'):
        n_pos = int(y.sum())
        prev = y.mean() * 100  # 转成百分比
        q_diseases.append({
            'code': disease,
            'n_samples': len(y),
            'n_pos': n_pos,
            'prevalence': prev
        })

# 按prevalence排序
q_diseases.sort(key=lambda x: x['prevalence'], reverse=True)

print("\n" + "="*70)
print("Q diseases sorted by prevalence:")
print("="*70)
print(f"{'Code':<10} {'N_samples':<12} {'N_positives':<12} {'Prevalence':<12}")
print("-"*70)

for d in q_diseases:
    print(f"{d['code']:<10} {d['n_samples']:<12} {d['n_pos']:<12} {d['prevalence']:.4f}%")

print("="*70)
print(f"\nTotal Q diseases found: {len(q_diseases)}")

# 保存结果（CSV格式，手动写）
with open('q_diseases_stats.csv', 'w') as f:
    f.write("code,n_samples,n_pos,prevalence_percent\n")
    for d in q_diseases:
        f.write(f"{d['code']},{d['n_samples']},{d['n_pos']},{d['prevalence']:.6f}\n")

print("✅ Saved to q_diseases_stats.csv")
