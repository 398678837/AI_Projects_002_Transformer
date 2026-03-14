"""
多头注意力演示
Multi-Head Attention
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("多头注意力演示")
print("=" * 70)

# 1. 多头注意力概念
print("\n1. 多头注意力概念...")

print("""
多头注意力 (Multi-Head Attention):
- 多个注意力头并行计算
- 每个头关注不同的信息
- 捕获多种语义关系
""")

# 2. 多头注意力计算
print("\n2. 多头注意力计算...")

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def multi_head_attention(Q, K, V, num_heads=4):
    d_model = Q.shape[-1]
    d_k = d_model // num_heads
    
    Q_reshaped = Q.reshape(Q.shape[0], num_heads, d_k)
    K_reshaped = K.reshape(K.shape[0], num_heads, d_k)
    V_reshaped = V.reshape(V.shape[0], num_heads, d_k)
    
    scores = np.matmul(Q_reshaped, K_reshaped.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, V_reshaped)
    
    output = output.reshape(Q.shape[0], d_model)
    return output, attention_weights

np.random.seed(42)
seq_len = 4
d_model = 8

Q = np.random.randn(seq_len, d_model)
K = np.random.randn(seq_len, d_model)
V = np.random.randn(seq_len, d_model)

output, weights = multi_head_attention(Q, K, V, num_heads=4)
print(f"  输出形状: {output.shape}")
print(f"  注意力头数: 4")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
for i in range(4):
    ax.plot(weights[0, i], label=f'头{i+1}', linewidth=2)
ax.set_xlabel('Key位置')
ax.set_ylabel('注意力权重')
ax.set_title('不同注意力头的权重')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
heads = ['头1\n句法', '头2\n语义', '头3\n位置', '头4\n关系']
importance = [0.85, 0.78, 0.72, 0.68]
ax.bar(heads, importance, color='steelblue', alpha=0.7)
ax.set_ylabel('重要性')
ax.set_title('不同注意力头的功能')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/multi_head_attention.png')
print("可视化已保存为 'images/multi_head_attention.png'")

# 4. 总结
print("\n" + "=" * 70)
print("多头注意力总结")
print("=" * 70)
print("""
多头注意力优势:
1. 捕获多种语义关系
2. 并行计算效率高
3. 模型表达能力更强
""")
print("=" * 70)
print("\nMulti-Head Attention Demo完成！")
