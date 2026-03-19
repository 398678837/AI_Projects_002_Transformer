"""
Prompt基础演示
Prompt Engineering Basics
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Prompt Engineering Basics Demo")
print("=" * 70)

# 1. Prompt基础
print("\n1. Prompt Engineering Basics...")

print("""
Prompt Engineering:
- 清晰明确：指令要清晰，避免歧义
- 结构化：使用统一格式
- 具体详细：提供详细要求
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
types = ['零样本', '单样本', '少样本']
accuracy = [65, 75, 85]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(types, accuracy, color=colors, alpha=0.7)
ax.set_xlabel('Prompt类型', fontsize=10)
ax.set_ylabel('准确率(%)', fontsize=10)
ax.set_title('不同Prompt类型的准确率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc}%', ha='center', va='bottom', fontsize=9)

ax = axes[1]
principles = ['清晰明确', '结构化', '具体详细']
importance = [9, 8, 7]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(principles, importance, color=colors, alpha=0.7)
ax.set_xlabel('核心原则', fontsize=10)
ax.set_ylabel('重要性(1-10)', fontsize=10)
ax.set_title('Prompt核心原则重要性', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, imp in zip(bars, importance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{imp}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'prompt_basics.png'))
print("可视化已保存为 'images/prompt_basics.png'")

# 3. Prompt类型
print("\n3. Prompt Types...")

print("""
Prompt Types:
1. 零样本Prompt：
   - 直接给出任务
   - 不提供示例
   - 适用于简单任务

2. 单样本Prompt：
   - 提供1个示例
   - 适用于中等复杂任务
   - 比零样本更准确

3. 少样本Prompt（Few-Shot）：
   - 提供2-3个示例
   - 适用于复杂任务
   - 最准确的类型
""")

# 4. 最佳实践
print("\n4. Best Practices...")

print("""
Best Practices:
1. 指令设计：
   - 明确角色
   - 清晰任务
   - 详细要求
   - 具体输出

2. 示例设计：
   - 代表性
   - 多样性
   - 清晰格式

3. 格式设计：
   - 统一格式
   - 易于解析
   - 可扩展
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Prompt Engineering Applications:
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
print("Prompt Engineering Summary")
print("=" * 70)

print("""
Key Concepts:
1. Prompt工程是给AI清晰、结构化指令的技术
2. 核心原则：清晰明确、结构化、具体详细
3. 基本结构：角色、任务、要求、输出

Prompt Types:
- 零样本Prompt
- 单样本Prompt
- 少样本Prompt（Few-Shot）

Best Practices:
- 指令设计
- 示例设计
- 格式设计

Prompt Engineering Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Prompt Engineering Basics Demo completed!")
print("=" * 70)
