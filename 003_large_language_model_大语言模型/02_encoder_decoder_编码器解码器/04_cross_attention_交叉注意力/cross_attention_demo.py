"""
交叉注意力演示
Cross-Attention
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("交叉注意力演示")
print("=" * 70)

# 1. 交叉注意力概念
print("\n1. 交叉注意力概念...")

print("""
交叉注意力 (Cross-Attention):
- 解码器关注编码器输出
- 连接两个不同序列
- 机器翻译核心组件
""")

# 2. 交叉注意力计算
print("\n2. 交叉注意力计算...")

def cross_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    return np.matmul(attention_weights, V), attention_weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

np.random.seed(42)
seq_len_q = 4
seq_len_kv = 6
d_model = 8

Q = np.random.randn(seq_len_q, d_model)
K = np.random.randn(seq_len_kv, d_model)
V = np.random.randn(seq_len_kv, d_model)

output, weights = cross_attention(Q, K, V)

print(f"  Query形状: {Q.shape}")
print(f"  Key形状: {K.shape}")
print(f"  Value形状: {V.shape}")
print(f"  输出形状: {output.shape}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
im = ax.imshow(weights, cmap='Blues', aspect='auto')
ax.set_xlabel('Key位置(编码器)')
ax.set_ylabel('Query位置(解码器)')
ax.set_title('交叉注意力权重')
plt.colorbar(im, ax=ax)

ax = axes[1]
q_words = ['我', '爱', '学习']
kv_words = ['I', 'love', 'learning', 'Python', 'NLP', '今天']
ax.bar(range(len(kv_words)), weights[0], color='steelblue', alpha=0.7)
ax.set_xticks(range(len(kv_words)))
ax.set_xticklabels(kv_words, rotation=30, ha='right')
ax.set_ylabel('注意力权重')
ax.set_title('解码器位置对编码器输出的关注')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/cross_attention.png')
print("可视化已保存为 'images/cross_attention.png'")

# 4. 总结
print("\n" + "=" * 70)
print("交叉注意力总结")
print("=" * 70)
print("""
交叉注意力作用:
1. 连接编码器和解码器
2. 让解码器关注Input
3. 机器翻译核心机制
""")
print("=" * 70)
print("\nCross-Attention Demo完成！")
