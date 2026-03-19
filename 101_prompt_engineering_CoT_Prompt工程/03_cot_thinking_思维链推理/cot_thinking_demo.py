"""
思维链推理演示
Chain of Thought (CoT) Thinking
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Chain of Thought (CoT) Thinking Demo")
print("=" * 70)

# 1. CoT思维链
print("\n1. Chain of Thought (CoT)...")

print("""
Chain of Thought (CoT):
- 先写思考过程
- 再给最终答案
- 大幅提升推理准确率
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = ['直接回答', 'CoT思维链']
accuracy = [70, 90]
colors = ['steelblue', 'coral']
bars = ax.bar(methods, accuracy, color=colors, alpha=0.7)
ax.set_xlabel('推理方法', fontsize=10)
ax.set_ylabel('准确率(%)', fontsize=10)
ax.set_title('不同推理方法的准确率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc}%', ha='center', va='bottom', fontsize=9)

ax = axes[1]
tasks = ['数学题', '逻辑题', '常识题', '编程题']
cot_accuracy = [95, 90, 85, 80]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(tasks, cot_accuracy, color=colors, alpha=0.7)
ax.set_xlabel('任务类型', fontsize=10)
ax.set_ylabel('准确率(%)', fontsize=10)
ax.set_title('CoT思维链在不同任务中的准确率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, acc in zip(bars, cot_accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'cot_thinking.png'))
print("可视化已保存为 'images/cot_thinking.png'")

# 3. CoT类型
print("\n3. CoT Types...")

print("""
CoT Types:
1. 简单CoT：
   - 直接给出思考过程
   - 适用于简单任务
   - 准确率提升：20%

2. 分步CoT：
   - 分步骤写出思考过程
   - 适用于中等复杂任务
   - 准确率提升：25%

3. 详细CoT：
   - 详细写出每个步骤
   - 适用于复杂任务
   - 准确率提升：30%
""")

# 4. 最佳实践
print("\n4. Best Practices...")

print("""
Best Practices:
1. 思考过程设计：
   - 分步骤：分步骤写出思考过程
   - 详细：详细写出每个步骤
   - 逻辑清晰：逻辑要清晰

2. 问题类型：
   - 数学题：适合CoT
   - 逻辑题：适合CoT
   - 常识题：适合CoT

3. 输出格式：
   - 统一格式：统一思考过程格式
   - 清晰易读：格式要清晰易读
   - 易于解析：格式要易于解析
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
CoT Thinking Applications:
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
print("CoT Thinking Summary")
print("=" * 70)

print("""
Key Concepts:
1. CoT思维链是让AI先写思考过程，再给答案的方法
2. 核心优势：提升准确率、增强解释性、易于调试
3. 基本结构：问题、要求、思考过程、最终答案

CoT Types:
- 简单CoT
- 分步CoT
- 详细CoT

Best Practices:
- 思考过程设计
- 问题类型
- 输出格式

CoT Thinking Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("CoT Thinking Demo completed!")
print("=" * 70)
