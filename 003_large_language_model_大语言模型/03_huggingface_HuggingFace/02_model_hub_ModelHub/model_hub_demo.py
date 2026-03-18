"""
Model Hub演示
模型市场
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Model Hub Demo")
print("=" * 70)

# 1. Model Hub概念
print("\n1. Model Hub Concept...")

print("""
Model Hub:
- Model sharing platform
- Hosts pre-trained models
- Community contributions
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
categories = ['NLP', 'CV', 'Audio', 'MLM', 'RL']
models_count = [50000, 30000, 15000, 25000, 8000]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(categories, models_count, color=colors, alpha=0.7)
ax.set_ylabel('Model Count', fontsize=10)
ax.set_title('Model Count by Category', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, models_count):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
            f'{count:,}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
downloads = [100, 85, 70, 60, 50]
wedges, texts, autotexts = ax.pie(downloads, labels=categories, autopct='%1.1f%%', startangle=90)
ax.set_title('Model Download Distribution', fontsize=12)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'model_hub.png'))
print("Visualization saved to 'images/model_hub.png'")

# 3. Model Hub特点
print("\n3. Model Hub Features...")

print("""
Model Hub Features:
1. Massive Models
   - 100,000+ pre-trained models
   - Cover multiple tasks and domains
   - Continuous growth

2. Easy to Use
   - Simple API
   - One-click loading
   - Compatible with Transformers

3. Community Driven
   - Open source community
   - Model quality assurance
   - Active developer community

4. Diverse Models
   - NLP models
   - CV models
   - Audio models
   - MLM models
   - RL models
""")

# 4. 应用场景
print("\n4. Applications...")

print("""
Model Hub Applications:
1. Research:
   - Quick experiments
   - Model comparison
   - Baseline models

2. Development:
   - Rapid prototyping
   - Production deployment
   - Model integration

3. Education:
   - Teaching examples
   - Project reference
   - Learning resources
""")

# 5. 总结
print("\n" + "=" * 70)
print("Model Hub Summary")
print("=" * 70)

print("""
Key Concepts:
1. Model Hub is Hugging Face's model sharing platform
2. Hosts 100,000+ pre-trained models
3. Easy to use with simple API
4. Community driven
5. Diverse models

Model Hub Features:
- Massive models
- Easy to use
- Community driven
- Diverse models

Model Hub Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Model Hub Demo completed!")
print("=" * 70)
