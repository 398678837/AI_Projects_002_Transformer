"""
Model Hub演示
模型市场
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Model Hub演示")
print("=" * 70)

# 1. Model Hub概念
print("\n1. Model Hub概念...")

print("""
Model Hub:
- 模型共享平台
- 托管预训练模型
- 社区贡献
""")

# 2. 可视化
print("\n2. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
categories = ['NLP', 'CV', 'Audio', 'MLM', 'RL']
models_count = [50000, 30000, 15000, 25000, 8000]
ax.bar(categories, models_count, color='steelblue', alpha=0.7)
ax.set_ylabel('模型数量')
ax.set_title('Model Hub各类模型数量')
ax.grid(True, alpha=0.3)

ax = axes[1]
downloads = [100, 85, 70, 60, 50]
ax.pie(downloads, labels=categories, autopct='%1.1f%%', startangle=90)
ax.set_title('各类模型下载占比')

plt.tight_layout()
plt.savefig('images/model_hub.png')
print("可视化已保存为 'images/model_hub.png'")

print("\n" + "=" * 70)
print("Model Hub总结")
print("=" * 70)
print("Model Hub是模型共享平台。")
print("=" * 70)
