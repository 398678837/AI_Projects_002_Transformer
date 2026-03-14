"""
Dataset Hub演示
数据集市场
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Dataset Hub演示")
print("=" * 70)

# 1. Dataset Hub概念
print("\n1. Dataset Hub概念...")

print("""
Dataset Hub:
- 数据集共享平台
- 托管各种数据集
- 数据预处理
""")

# 2. 可视化
print("\n2. 可视化...")

fig, ax = plt.subplots(figsize=(10, 5))

categories = ['NLP', 'CV', 'Audio', 'Tabular', 'RL']
datasets_count = [8000, 5000, 3000, 4000, 1500]
ax.bar(categories, datasets_count, color='steelblue', alpha=0.7)
ax.set_ylabel('数据集数量')
ax.set_title('Dataset Hub各类数据集数量')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/dataset_hub.png')
print("可视化已保存为 'images/dataset_hub.png'")

print("\n" + "=" * 70)
print("Dataset Hub总结")
print("=" * 70)
print("Dataset Hub是数据集共享平台。")
print("=" * 70)
