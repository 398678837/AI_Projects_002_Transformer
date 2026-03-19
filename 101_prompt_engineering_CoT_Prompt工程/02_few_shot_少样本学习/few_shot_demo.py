"""
少样本学习演示
Few-Shot Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Few-Shot Learning Demo")
print("=" * 70)

# 1. Few-Shot学习
print("\n1. Few-Shot Learning...")

print("""
Few-Shot Learning:
- 提供2-3个示例
- AI照着格式输出
- 无需训练，直接使用
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = ['零样本', '单样本', '少样本', '多样本']
accuracy = [65, 75, 85, 90]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(methods, accuracy, color=colors, alpha=0.7)
ax.set_xlabel('方法', fontsize=10)
ax.set_ylabel('准确率(%)', fontsize=10)
ax.set_title('不同方法的准确率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc}%', ha='center', va='bottom', fontsize=9)

ax = axes[1]
examples = ['1个示例', '2个示例', '3个示例', '4个示例']
accuracy = [75, 85, 88, 90]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(examples, accuracy, color=colors, alpha=0.7)
ax.set_xlabel('示例数量', fontsize=10)
ax.set_ylabel('准确率(%)', fontsize=10)
ax.set_title('不同示例数量的准确率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'few_shot.png'))
print("可视化已保存为 'images/few_shot.png'")

# 3. Few-Shot类型
print("\n3. Few-Shot Types...")

print("""
Few-Shot Types:
1. 单样本（One-Shot）：
   - 提供1个示例
   - 适用于简单任务
   - 准确率：75%

2. 少样本（Few-Shot）：
   - 提供2-3个示例
   - 适用于中等复杂任务
   - 准确率：85%

3. 多样本（Many-Shot）：
   - 提供多个示例
   - 适用于复杂任务
   - 准确率：90%
""")

# 4. 最佳实践
print("\n4. Best Practices...")

print("""
Best Practices:
1. 示例设计：
   - 代表性：示例要能代表典型情况
   - 多样性：覆盖不同情况
   - 清晰格式：格式要清晰易懂

2. 示例数量：
   - 2-3个示例最佳
   - 过多会增加成本
   - 过少效果不佳

3. 示例顺序：
   - 从简单到复杂
   - 从典型到特殊
   - 保持一致性
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Few-Shot Learning Applications:
1. Research:
   - 快速实验
   - 模型比较
   - 基线模型

2. Development:
   - 快速原型
   - 生产部署
   - 模型集成

3. Education:
   - 教学示例
   - 项目参考
   - 学习资源
""")

# 6. 总结
print("\n" + "=" * 70)
print("Few-Shot Learning Summary")
print("=" * 70)

print("""
Key Concepts:
1. Few-Shot学习是给AI 2-3个例子，让AI照着格式输出的方法
2. 核心优势：快速适应、格式一致、准确率高
3. 基本结构：任务、示例1、示例2、请按示例格式输出

Few-Shot Types:
- 单样本（One-Shot）
- 少样本（Few-Shot）
- 多样本（Many-Shot）

Best Practices:
- 示例设计
- 示例数量
- 示例顺序

Few-Shot Learning Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Few-Shot Learning Demo completed!")
print("=" * 70)
