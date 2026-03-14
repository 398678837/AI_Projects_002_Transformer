"""
Qwen模型加载演示
Model Loading
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Qwen模型加载演示")
print("=" * 70)

# 1. 模型加载
print("\n1. 模型加载...")

print("""
Qwen模型加载:
- 使用AutoModelForCausalLM
- 加载量化模型
- 配置生成参数
""")

# 2. 可视化
print("\n2. 可视化...")

fig, ax = plt.subplots(figsize=(10, 5))

models = ['Qwen-0.5B', 'Qwen-1.8B', 'Qwen-7B', 'Qwen-14B']
params = [0.5, 1.8, 7, 14]
gpu_memory = [1, 4, 14, 28]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, params, width, label='参数量(B)', color='steelblue', alpha=0.7)
ax.bar(x + width/2, gpu_memory, width, label='显存需求(GB)', color='coral', alpha=0.7)
ax.set_ylabel('数值')
ax.set_title('Qwen各版本资源需求')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/qwen_model_loading.png')
print("可视化已保存为 'images/qwen_model_loading.png'")

print("\n" + "=" * 70)
print("Qwen模型加载总结")
print("=" * 70)
print("Qwen模型从Hugging Face加载。")
print("=" * 70)
