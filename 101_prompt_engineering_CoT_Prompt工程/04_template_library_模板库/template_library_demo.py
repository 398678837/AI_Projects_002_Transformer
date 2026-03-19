"""
模板库演示
Prompt Template Library
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Prompt Template Library Demo")
print("=" * 70)

# 1. 模板库
print("\n1. Template Library...")

print("""
Template Library:
- 通用对话模板
- 代码解释/优化模板
- 文本总结模板
- 逻辑推理模板
- 数据处理模板
- Few-Shot模板
- CoT模板
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
templates = ['通用对话', '代码', '总结', '推理', '数据处理']
usage = [30, 20, 15, 20, 15]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(templates, usage, color=colors, alpha=0.7)
ax.set_xlabel('模板类型', fontsize=10)
ax.set_ylabel('使用频率(%)', fontsize=10)
ax.set_title('不同模板类型的使用频率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, use in zip(bars, usage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{use}%', ha='center', va='bottom', fontsize=9)

ax = axes[1]
templates = ['通用对话', '代码', '总结', '推理', '数据处理']
satisfaction = [85, 80, 88, 90, 75]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(templates, satisfaction, color=colors, alpha=0.7)
ax.set_xlabel('模板类型', fontsize=10)
ax.set_ylabel('满意度(%)', fontsize=10)
ax.set_title('不同模板类型的满意度', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, sat in zip(bars, satisfaction):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{sat}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'template_library.png'))
print("可视化已保存为 'images/template_library.png'")

# 3. 模板类型
print("\n3. Template Types...")

print("""
Template Types:
1. 通用对话模板：
   - 用于日常对话
   - 使用频率：30%
   - 满意度：85%

2. 代码模板：
   - 用于代码解释/优化
   - 使用频率：20%
   - 满意度：80%

3. 总结模板：
   - 用于文本总结
   - 使用频率：15%
   - 满意度：88%

4. 推理模板：
   - 用于逻辑推理
   - 使用频率：20%
   - 满意度：90%

5. 数据处理模板：
   - 用于数据处理
   - 使用频率：15%
   - 满意度：75%
""")

# 4. 最佳实践
print("\n4. Best Practices...")

print("""
Best Practices:
1. 模板设计：
   - 通用性：模板要通用
   - 可扩展：模板要可扩展
   - 易用性：模板要易用

2. 模板分类：
   - 按场景分类
   - 按任务分类
   - 按格式分类

3. 模板使用：
   - 选择合适模板
   - 修改参数
   - 测试效果
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Template Library Applications:
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
print("Template Library Summary")
print("=" * 70)

print("""
Key Concepts:
1. 模板库是预定义的Prompt结构集合
2. 核心优势：快速使用、统一格式、高质量
3. 常用模板：通用对话、代码、总结、推理、数据处理、Few-Shot、CoT

Template Types:
- 通用对话模板
- 代码模板
- 总结模板
- 推理模板
- 数据处理模板

Best Practices:
- 模板设计
- 模板分类
- 模板使用

Template Library Applications:
- Research
- Development
- Education
""")

print("=" * 70)
print("Template Library Demo completed!")
print("=" * 70)
