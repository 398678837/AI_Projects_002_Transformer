"""
CoT思维链推理Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("CoT思维链推理Demo")
print("=" * 70)

# 1. CoT推理
print("\n1. CoT推理示例...")

print("""
CoT思维链推理:
- 先写思考过程
- 再给最终答案
- 大幅提升推理准确率
""")

# 2. 数学题推理
print("\n2. 数学题推理...")

print("""
问题：小明有5个苹果，吃了2个，又买了3个，现在有多少个苹果？

思考过程：
1. 初始有5个苹果
2. 吃了2个，剩余5-2=3个
3. 又买了3个，现在有3+3=6个

最终答案：6
""")

# 3. 逻辑题推理
print("\n3. 逻辑题推理...")

print("""
问题：如果所有的猫都是动物，有些动物是狗，那么所有的猫都是狗吗？

思考过程：
1. 所有的猫都是动物
2. 有些动物是狗
3. 但这并不意味着所有的猫都是狗
4. 猫和狗都是动物，但属于不同种类

最终答案：不是
""")

# 4. 可视化
print("\n4. 可视化...")

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
plt.savefig(os.path.join(images_dir, 'cot_accuracy.png'))
print("可视化已保存为 'images/cot_accuracy.png'")

# 5. 总结
print("\n" + "=" * 70)
print("CoT思维链总结")
print("=" * 70)

print("""
CoT思维链核心:
1. 先写思考过程
2. 再给最终答案
3. 大幅提升推理准确率

CoT思维链应用场景:
- 数学题
- 逻辑题
- 常识题
- 编程题

CoT思维链优势:
- 更高准确率
- 更好解释性
- 更易调试

CoT思维链局限性:
- 更长生成时间
- 更多计算资源
- 可能产生错误思考过程
""")

print("=" * 70)
print("CoT思维链Demo完成!")
print("=" * 70)
