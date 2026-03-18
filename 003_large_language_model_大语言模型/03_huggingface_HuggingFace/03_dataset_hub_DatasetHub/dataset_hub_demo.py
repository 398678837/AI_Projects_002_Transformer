"""
Dataset Hub演示
数据集市场
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Dataset Hub Demo")
print("=" * 70)

# 1. Dataset Hub概念
print("\n1. Dataset Hub Concept...")

print("""
Dataset Hub:
- Dataset sharing platform
- Hosts various datasets
- Data preprocessing
""")

# 2. 可视化
print("\n2. Visualization...")

fig, ax = plt.subplots(figsize=(12, 5))

categories = ['NLP', 'CV', 'Audio', 'Tabular', 'RL']
datasets_count = [8000, 5000, 3000, 4000, 1500]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(categories, datasets_count, color=colors, alpha=0.7)
ax.set_ylabel('Dataset Count', fontsize=10)
ax.set_title('Dataset Count by Category', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, datasets_count):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
            f'{count:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'dataset_hub.png'))
print("Visualization saved to 'images/dataset_hub.png'")

# 3. Dataset Hub特点
print("\n3. Dataset Hub Features...")

print("""
Dataset Hub Features:
1. Massive Datasets
   - 100,000+ datasets
   - Cover multiple tasks and domains
   - Continuous growth

2. Easy to Use
   - Simple API
   - One-click loading
   - Compatible with Transformers

3. Community Driven
   - Open source community
   - Dataset quality assurance
   - Active developer community

4. Diverse Datasets
   - NLP datasets
   - CV datasets
   - Audio datasets
   - Tabular datasets
   - RL datasets
""")

# 4. 应用场景
print("\n4. Applications...")

print("""
Dataset Hub Applications:
1. Research:
   - Quick experiments
   - Dataset comparison
   - Baseline models

2. Development:
   - Rapid prototyping
   - Production deployment
   - Dataset integration

3. Education:
   - Teaching examples
   - Project reference
   - Learning resources
""")

# 5. 总结
print("\n" + "=" * 70)
print("Dataset Hub Summary")
print("=" * 70)

print("""
Key Concepts:
1. Dataset Hub is Hugging Face's dataset sharing platform
2. Hosts 100,000+ datasets
3. Easy to use with simple API
4. Community driven
5. Diverse datasets

Dataset Hub Features:
- Massive datasets
- Easy to use
- Community driven
- Diverse datasets

Dataset Hub Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Dataset Hub Demo completed!")
print("=" * 70)
