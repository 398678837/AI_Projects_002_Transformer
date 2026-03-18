"""
预训练模型演示
BERT、GPT等
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("预训练模型演示")
print("=" * 70)

# 1. 预训练概念
print("\n1. 预训练概念...")

print("""
预训练 (Pre-training):
- 在大规模数据上预训练
- 学习通用语言知识
- 迁移到下游任务

优势:
- 减少标注数据需求
- 提高下游任务性能
- 加速训练
""")

# 2. BERT vs GPT
print("\n2. BERT vs GPT对比...")

models = {
    'BERT': {
        'architecture': 'Transformer Encoder',
        'training': 'Masked LM',
        'direction': '双向',
        'tasks': '理解为主'
    },
    'GPT': {
        'architecture': 'Transformer Decoder',
        'training': 'Causal LM',
        'direction': '单向',
        'tasks': '生成为主'
    },
    'T5': {
        'architecture': 'Encoder-Decoder',
        'training': 'Text-to-Text',
        'direction': '双向',
        'tasks': '统一框架'
    }
}

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
model_names = ['BERT', 'GPT-2', 'GPT-3', 'GPT-4', 'T5']
params = [110, 1247, 175000, 1800000, 220]
params_billion = [p/1000 if p > 100 else p for p in params]

bars = ax.bar(model_names, params_billion, color=['steelblue', 'coral', 'green', 'purple', 'orange'], alpha=0.7)
ax.set_ylabel('参数 (百万/十亿)')
ax.set_title('预训练模型参数量')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

for bar, p in zip(bars, params):
    if p >= 1000:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{p/1000:.0f}B', ha='center', fontsize=9)
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{p}M', ha='center', fontsize=9)

ax = axes[1]
ax.axis('off')
table_data = [
    ['模型', '架构', '预训练任务', '方向'],
    ['BERT', 'Encoder', 'MLM', '双向'],
    ['GPT', 'Decoder', 'CLM', '单向'],
    ['T5', 'Enc-Dec', 'T2T', '双向']
]

y_pos = 0.9
for row in table_data:
    x_pos = 0.1
    for cell in row:
        ax.text(x_pos, y_pos, cell, fontsize=12, 
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat' if row == table_data[0] else 'white'))
        x_pos += 0.25
    y_pos -= 0.15

ax.set_title('预训练模型对比', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'pretraining.png'))
print("可视化已保存为 'images/pretraining.png'")

# 4. 总结
print("\n" + "=" * 70)
print("预训练模型总结")
print("=" * 70)
print("""
| 模型 | 架构 | 预训练任务 | 特点 |
|------|------|------------|------|
| BERT | Encoder | MLM | 理解 |
| GPT | Decoder | CLM | 生成 |
| T5 | Enc-Dec | Text-to-Text | 统一 |

预训练+微调范式:
1. 大规模预训练
2. 任务微调
3. 下游应用
""")
print("=" * 70)
print("\nPre-training Demo完成！")
