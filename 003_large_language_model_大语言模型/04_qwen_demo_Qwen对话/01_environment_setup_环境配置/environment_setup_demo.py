"""
Qwen环境配置演示
Environment Setup
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Qwen Environment Setup Demo")
print("=" * 70)

# 1. 环境配置
print("\n1. Environment Setup...")

print("""
Environment Setup:
- Python >= 3.8
- PyTorch >= 1.8
- transformers library
- CUDA (GPU support)
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
requirements = ['Python', 'PyTorch', 'Transformers', 'CUDA', 'GPU Memory']
scores = [95, 90, 88, 85, 80]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(requirements, scores, color=colors, alpha=0.7)
ax.set_ylabel('Importance Score', fontsize=10)
ax.set_title('Qwen Environment Setup Requirements', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{score}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
versions = ['GPU', 'RAM', 'Disk', 'Python', 'PyTorch']
requirements = [6, 16, 50, 3.8, 1.8]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(versions, requirements, color=colors, alpha=0.7)
ax.set_ylabel('Minimum Requirement', fontsize=10)
ax.set_title('Qwen Environment Requirements', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, req in zip(bars, requirements):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{req}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'environment_setup.png'))
print("Visualization saved to 'images/environment_setup.png'")

# 3. 环境配置总结
print("\n3. Environment Setup Summary...")

print("""
Environment Setup Summary:
1. Hardware Requirements:
   - GPU: NVIDIA GPU with >= 6GB VRAM
   - RAM: >= 16GB
   - Disk: >= 50GB

2. Software Requirements:
   - Python >= 3.8
   - PyTorch >= 1.8
   - transformers library
   - CUDA (for GPU support)

3. Installation:
   - pip install torch transformers
   - pip install accelerate (for quantization)
   - pip install bitsandbytes (for 4-bit/8-bit quantization)
""")

# 4. 应用场景
print("\n4. Applications...")

print("""
Environment Setup Applications:
1. Research:
   - Quick experiments
   - Model comparison
   - Baseline models

2. Development:
   - Quick prototyping
   - Production deployment
   - Model integration

3. Education:
   - Teaching examples
   - Project reference
   - Learning resources
""")

# 5. 总结
print("\n" + "=" * 70)
print("Environment Setup Summary")
print("=" * 70)

print("""
Key Concepts:
1. Environment setup is the first step in using Qwen
2. Hardware requirements: GPU, RAM, Disk
3. Software requirements: Python, PyTorch, transformers
4. Installation: pip install torch transformers

Environment Setup Requirements:
- Hardware: GPU, RAM, Disk
- Software: Python, PyTorch, transformers

Environment Setup Applications:
- Research
- Development
- Education

Installation Steps:
1. Install Python >= 3.8
2. Install PyTorch >= 1.8
3. Install transformers library
4. Install optional packages (accelerate, bitsandbytes)
""")

print("=" * 70)
print("Environment Setup Demo completed!")
print("=" * 70)
