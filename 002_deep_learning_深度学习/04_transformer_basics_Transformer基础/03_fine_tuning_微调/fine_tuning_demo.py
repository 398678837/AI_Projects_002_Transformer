"""
微调演示
下游任务适配
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("微调演示")
print("=" * 70)

# 1. 微调概念
print("\n1. 微调概念...")

print("""
微调 (Fine-tuning):
- 加载预训练模型
- 在下游数据上继续训练
- 适配特定任务

常见策略:
1. 全参数微调: 更新所有参数
2. 冻结+微调: 冻结部分层
3. LoRA: 低秩适配
4. Prefix-Tuning: 前缀调整
""")

# 2. 微调策略对比
print("\n2. 微调策略...")

strategies = {
    '全参数微调': {'params': 100, 'accuracy': 95, 'time': 10},
    '冻结Encoder': {'params': 20, 'accuracy': 88, 'time': 3},
    'LoRA': {'params': 1, 'accuracy': 92, 'time': 2},
    'Prefix-Tuning': {'params': 0.5, 'accuracy': 90, 'time': 1}
}

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

names = list(strategies.keys())
params = [strategies[n]['params'] for n in names]
acc = [strategies[n]['accuracy'] for n in names]
time_ = [strategies[n]['time'] for n in names]

ax = axes[0]
bars = ax.bar(names, params, color='steelblue', alpha=0.7)
ax.set_ylabel('参数量 (%)')
ax.set_title('微调参数量对比')
ax.set_xticklabels(names, rotation=15)
ax.grid(True, alpha=0.3)

ax = axes[1]
bars = ax.bar(names, acc, color='coral', alpha=0.7)
ax.set_ylabel('Accuracy (%)')
ax.set_title('微调Accuracy对比')
ax.set_xticklabels(names, rotation=15)
ax.set_ylim(80, 100)
ax.grid(True, alpha=0.3)

ax = axes[2]
bars = ax.bar(names, time_, color='green', alpha=0.7)
ax.set_ylabel('训练时间')
ax.set_title('微调时间对比')
ax.set_xticklabels(names, rotation=15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'fine_tuning.png'))
print("可视化已保存为 'images/fine_tuning.png'")

# 4. 总结
print("\n" + "=" * 70)
print("微调总结")
print("=" * 70)
print("""
| 策略 | 参数量 | Accuracy | 速度 |
|------|--------|--------|------|
| 全参数 | 100% | 最高 | 最慢 |
| 冻结 | 20% | 中等 | 快 |
| LoRA | 1% | 接近最高 | 最快 |

选择策略:
- 数据少: LoRA, Prefix
- 数据多: 全参数微调
""")
print("=" * 70)
print("\nFine-tuning Demo完成！")
