"""
注意力可视化演示
Attention Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("注意力可视化演示")
print("=" * 70)

# 1. 注意力可视化概念
print("\n1. 注意力可视化概念...")

print("""
注意力可视化:
- 展示注意力权重分布
- 理解模型关注哪些位置
- 辅助模型分析和调试
""")

# 2. 可视化
print("\n2. 注意力可视化...")

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sentence1 = "The cat sat on the mat"
sentence2 = "猫 坐在 垫子 上"
words1 = sentence1.split()
words2 = sentence2.split()

attention_matrix = np.random.rand(len(words1), len(words2))
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

ax = axes[0, 0]
im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(np.arange(len(words2)))
ax.set_yticks(np.arange(len(words1)))
ax.set_xticklabels(words2)
ax.set_yticklabels(words1)
ax.set_xlabel('Key')
ax.set_ylabel('Query')
ax.set_title('中英文注意力')
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
for i, word in enumerate(words1):
    ax.bar(range(len(words2)), attention_matrix[i], label=word, alpha=0.7)
ax.set_xlabel('Key位置')
ax.set_ylabel('注意力权重')
ax.set_title('各Query的注意力分布')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

ax = axes[1, 0]
layer_weights = np.random.rand(6, 6)
im = ax.imshow(layer_weights, cmap='hot', aspect='auto')
ax.set_xlabel('Key位置')
ax.set_ylabel('Layer层数')
ax.set_title('不同层的注意力模式')
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
head_importance = [0.85, 0.72, 0.68, 0.55, 0.48, 0.42]
heads = [f'头{i+1}' for i in range(6)]
ax.bar(heads, head_importance, color='steelblue', alpha=0.7)
ax.set_ylabel('重要性')
ax.set_title('不同注意力头的重要性')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/attention_visualization.png')
print("可视化已保存为 'images/attention_visualization.png'")

# 3. 总结
print("\n" + "=" * 70)
print("注意力可视化总结")
print("=" * 70)
print("""
注意力可视化应用:
1. 理解模型行为
2. 分析语义对齐
3. 调试模型问题
4. 解释模型决策
""")
print("=" * 70)
print("\nAttention Visualization Demo完成！")
