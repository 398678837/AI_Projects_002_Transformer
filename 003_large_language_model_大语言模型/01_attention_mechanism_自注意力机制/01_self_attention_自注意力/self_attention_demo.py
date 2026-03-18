"""
自注意力机制演示
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Self-Attention Mechanism Demo")
print("=" * 70)

# 1. 定义自注意力计算
def scaled_dot_product_attention(Q, K, V, mask=None):
    """计算缩放点积注意力"""
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    attention_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # 应用掩码
    if mask is not None:
        attention_scores = np.where(mask == 0, -np.inf, attention_scores)
    
    # 计算注意力权重
    attention_weights = softmax(attention_scores)
    
    # 计算加权和
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x):
    """计算softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 2. 示例：句子编码
print("\n1. 示例：句子编码")

# 假设有3个词，每个词的嵌入维度为4
word_embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4],  # 词1
    [0.5, 0.6, 0.7, 0.8],  # 词2
    [0.9, 1.0, 1.1, 1.2]   # 词3
])

# 定义投影矩阵（实际中是可学习的参数）
d_model = 4
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)

# 计算Q, K, V
Q = np.matmul(word_embeddings, W_q)
K = np.matmul(word_embeddings, W_k)
V = np.matmul(word_embeddings, W_v)

# 计算自注意力
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"输入词嵌入形状: {word_embeddings.shape}")
print(f"Q/K/V形状: {Q.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
print(f"输出形状: {output.shape}")

print("\n注意力权重:")
print(np.round(attention_weights, 3))

print("\n输出向量:")
print(np.round(output, 3))

# 3. 可视化注意力权重
print("\n2. 可视化注意力权重")

plt.figure(figsize=(8, 6))
plt.imshow(attention_weights, cmap='viridis')
plt.colorbar()
plt.title('Self-Attention Weights')
plt.xlabel('Key positions')
plt.ylabel('Query positions')
plt.xticks([0, 1, 2], ['Word 1', 'Word 2', 'Word 3'])
plt.yticks([0, 1, 2], ['Word 1', 'Word 2', 'Word 3'])

# 添加数值标签
for i in range(attention_weights.shape[0]):
    for j in range(attention_weights.shape[1]):
        plt.text(j, i, f'{attention_weights[i, j]:.3f}', 
                 ha='center', va='center', color='white')

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'self_attention_weights.png'))
print("注意力权重可视化已保存到 'images/self_attention_weights.png'")

# 4. 自注意力的优势
print("\n3. 自注意力的优势")
print("- 并行计算效率高")
print("- 捕获长距离依赖")
print("- 可解释性强")
print("- 适应不同长度的序列")

# 5. 与其他注意力机制对比
print("\n4. 与其他注意力机制对比")
print("| 注意力类型 | 计算复杂度 | 长距离依赖 | 并行性 |")
print("|-----------|------------|------------|--------|")
print("| 自注意力 | O(n²d) | 优秀 | 优秀 |")
print("| 循环注意力 | O(nd) | 较差 | 较差 |")
print("| 卷积注意力 | O(ndk) | 中等 | 中等 |")

print("\n" + "=" * 70)
print("Self-Attention Demo completed!")
print("=" * 70)
