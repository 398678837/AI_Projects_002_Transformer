"""
数学逻辑推理演示
Math and Logic Reasoning
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Math and Logic Reasoning Demo")
print("=" * 70)

# 1. 数学逻辑推理
print("\n1. Math and Logic Reasoning...")

print("""
Math and Logic Reasoning:
- 数学推理：解决数学题
- 逻辑推理：解决逻辑题
- 常识推理：解决常识题
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
plt.savefig(os.path.join(images_dir, 'math_logic.png'))
print("可视化已保存为 'images/math_logic.png'")

# 3. 推理类型
print("\n3. Reasoning Types...")

print("""
Reasoning Types:
1. 数学推理：
   - 解决数学题
   - 准确率：95%
   - 适用场景：数学计算、代数、几何

2. 逻辑推理：
   - 解决逻辑题
   - 准确率：90%
   - 适用场景：逻辑判断、推理、证明

3. 常识推理：
   - 解决常识题
   - 准确率：85%
   - 适用场景：常识判断、推理、应用
""")

# 4. 最佳实践
print("\n4. Best Practices...")

print("""
Best Practices:
1. 推理过程设计：
   - 分步骤：分步骤写出推理过程
   - 详细：详细写出每个步骤
   - 逻辑清晰：逻辑要清晰

2. 问题类型：
   - 数学题：适合CoT
   - 逻辑题：适合CoT
   - 常识题：适合CoT

3. 输出格式：
   - 统一格式：统一推理过程格式
   - 清晰易读：格式要清晰易读
   - 易于解析：格式要易于解析
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Math and Logic Reasoning Applications:
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
print("Math and Logic Reasoning Summary")
print("=" * 70)

print("""
Key Concepts:
1. 数学逻辑推理是使用CoT思维链解决数学题和逻辑题的方法
2. 核心优势：提升准确率、增强解释性、易于调试
3. 基本结构：问题、要求、思考过程、最终答案

Reasoning Types:
- 数学推理
- 逻辑推理
- 常识推理

Best Practices:
- 推理过程设计
- 问题类型
- 输出格式

Math and Logic Reasoning Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Math and Logic Reasoning Demo completed!")
print("=" * 70)
