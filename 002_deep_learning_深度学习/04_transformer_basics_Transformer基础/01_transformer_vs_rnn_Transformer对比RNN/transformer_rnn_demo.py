"""
Transformer vs RNN对比演示
注意力机制
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Transformer vs RNN对比演示")
print("=" * 70)

# 1. 注意力机制概念
print("\n1. 注意力机制概念...")

print("""
注意力机制 (Attention):
- 模拟人脑注意力
- 动态分配计算资源
- 解决长距离依赖

核心思想:
- _query_: 我要找什么
- _key_: 哪些位置有相关信息
- _value_: 相关信息的内容
""")

# 2. 简单注意力计算
print("\n2. 注意力计算...")

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def attention(Q, K, V):
    d_k = Q.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    return np.dot(attention_weights, V), attention_weights

Q = np.random.randn(1, 4)
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

output, weights = attention(Q, K, V)
print(f"  Query: {Q.shape}")
print(f"  Keys: {K.shape}")
print(f"  Values: {V.shape}")
print(f"  输出: {output.shape}")
print(f"  注意力权重: {weights}")

# 3. Transformer vs RNN
print("\n3. Transformer vs RNN对比...")

comparison = {
    '并行计算': ['✓ 支持', '✗ 顺序计算'],
    '长距离依赖': ['✓ O(1)', '✗ O(n)'],
    '梯度流动': ['✓ 稳定', '✗ 梯度消失'],
    '计算复杂度': ['O(n²·d)', 'O(n·d²)'],
    '内存': ['高', '低']
}

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
attention_matrix = np.random.rand(8, 8)
attention_matrix = (attention_matrix + attention_matrix.T) / 2
np.fill_diagonal(attention_matrix, 0)
attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

im = ax.imshow(attention_matrix, cmap='Blues')
ax.set_xlabel('Key位置')
ax.set_ylabel('Query位置')
ax.set_title('注意力权重矩阵')
plt.colorbar(im, ax=ax)

ax = axes[1]
metrics = ['并行计算', '长距离依赖', '梯度流动', '计算效率']
rnn_scores = [2, 3, 3, 5]
transformer_scores = [9, 9, 8, 7]

x = np.arange(len(metrics))
width = 0.35

ax.bar(x - width/2, rnn_scores, width, label='RNN', color='steelblue', alpha=0.7)
ax.bar(x + width/2, transformer_scores, width, label='Transformer', color='coral', alpha=0.7)
ax.set_ylabel('评分 (1-10)')
ax.set_title('RNN vs Transformer 对比')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/transformer_vs_rnn.png')
print("可视化已保存为 'images/transformer_vs_rnn.png'")

# 5. 注意力可视化
fig, ax = plt.subplots(figsize=(10, 8))

words = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.', 'It', 'was', 'happy']
n = len(words)

attention_weights = np.random.rand(n, n)
attention_weights = (attention_weights + attention_weights.T) / 2
np.fill_diagonal(attention_weights, 0)
attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

im = ax.imshow(attention_weights, cmap='viridis')

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(words, rotation=45, ha='right')
ax.set_yticklabels(words)

for i in range(n):
    for j in range(n):
        text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                      ha='center', va='center', color='white' if attention_weights[i, j] > 0.3 else 'black', fontsize=8)

ax.set_title('自注意力机制可视化', fontsize=14)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/self_attention.png')
print("可视化已保存为 'images/self_attention.png'")

# 6. 总结
print("\n" + "=" * 70)
print("Transformer vs RNN总结")
print("=" * 70)
print("""
| 特性 | RNN | Transformer |
|------|-----|-------------|
| 并行计算 | 否 | 是 |
| 长距离依赖 | 梯度消失 | 注意力机制 |
| 计算效率 | O(n) | O(n²) |
| 内存 | 低 | 高 |

Transformer优势:
- 并行计算，速度快
- 注意力机制捕获长距离依赖
- 已成为NLP主流架构
""")
print("=" * 70)
print("\nTransformer vs RNN Demo完成！")
