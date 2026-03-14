"""
Qwen环境配置演示
Environment Setup
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Qwen环境配置演示")
print("=" * 70)

# 1. 环境配置
print("\n1. 环境配置...")

print("""
环境配置:
- Python >= 3.8
- PyTorch >= 1.8
- transformers库
- CUDA (GPU支持)
""")

# 2. 可视化
print("\n2. 可视化...")

fig, ax = plt.subplots(figsize=(10, 5))

requirements = ['Python', 'PyTorch', 'Transformers', 'CUDA', 'GPU Memory']
scores = [95, 90, 88, 85, 80]
ax.barh(requirements, scores, color='steelblue', alpha=0.7)
ax.set_xlabel('重要性评分')
ax.set_title('Qwen环境配置要素')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/environment_setup.png')
print("可视化已保存为 'images/environment_setup.png'")

print("\n" + "=" * 70)
print("环境配置总结")
print("=" * 70)
print("配置好环境后即可使用Qwen模型。")
print("=" * 70)
