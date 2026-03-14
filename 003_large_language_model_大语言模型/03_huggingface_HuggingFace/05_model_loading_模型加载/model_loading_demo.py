"""
模型加载演示
Model Loading
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("模型加载演示")
print("=" * 70)

# 1. 模型加载概念
print("\n1. 模型加载概念...")

print("""
模型加载:
- 从Hub加载预训练模型
- 指定模型名称和配置
- 自动下载模型权重
""")

# 2. 可视化
print("\n2. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
loading_times = ['BERT', 'GPT-2', 'T5', 'Llama-7B']
times = [5, 15, 20, 120]
ax.bar(loading_times, times, color='steelblue', alpha=0.7)
ax.set_ylabel('加载时间(秒)')
ax.set_title('不同模型加载时间')
ax.grid(True, alpha=0.3)

ax = axes[1]
memory = ['BERT', 'GPT-2', 'T5', 'Llama-7B']
ram = [500, 1000, 1500, 14000]
ax.bar(memory, ram, color='coral', alpha=0.7)
ax.set_ylabel('内存占用(MB)')
ax.set_title('不同模型内存占用')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/model_loading.png')
print("可视化已保存为 'images/model_loading.png'")

print("\n" + "=" * 70)
print("模型加载总结")
print("=" * 70)
print("模型加载是从Hub获取预训练模型的过程。")
print("=" * 70)
