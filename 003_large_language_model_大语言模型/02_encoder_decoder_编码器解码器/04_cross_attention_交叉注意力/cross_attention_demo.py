"""
交叉注意力演示
Cross-Attention Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Cross-Attention Demo")
print("=" * 70)

# 1. 交叉注意力概念
print("\n1. Cross-Attention Concept...")

print("""
Cross-Attention (交叉注意力):
- 解码器关注编码器输出
- 连接两个不同序列
- 机器翻译核心组件
""")

# 2. 交叉注意力计算
print("\n2. Cross-Attention Calculation...")

def cross_attention(Q, K, V):
    """计算交叉注意力"""
    d_k = Q.shape[-1]
    
    # 计算注意力分数
    attention_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # 计算注意力权重
    attention_weights = softmax(attention_scores)
    
    # 计算加权和
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights

def softmax(x):
    """softmax函数"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

np.random.seed(42)
seq_len_q = 4  # 解码器序列长度
seq_len_kv = 6  # 编码器序列长度
d_model = 8  # 模型维度

Q = np.random.randn(seq_len_q, d_model)  # 解码器查询
K = np.random.randn(seq_len_kv, d_model)  # 编码器键
V = np.random.randn(seq_len_kv, d_model)  # 编码器值

output, weights = cross_attention(Q, K, V)

print(f"  Query形状 (解码器): {Q.shape}")
print(f"  Key形状 (编码器): {K.shape}")
print(f"  Value形状 (编码器): {V.shape}")
print(f"  输出形状: {output.shape}")
print(f"  注意力权重形状: {weights.shape}")

# 3. 可视化
print("\n3. Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 交叉注意力权重热力图
ax = axes[0, 0]
im = ax.imshow(weights, cmap='viridis', aspect='auto')
ax.set_xlabel('Key positions (Encoder)', fontsize=10)
ax.set_ylabel('Query positions (Decoder)', fontsize=10)
ax.set_title('Cross-Attention Weights', fontsize=12)
plt.colorbar(im, ax=ax)

# 添加数值标签
for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        ax.text(j, i, f'{weights[i, j]:.2f}', 
                ha='center', va='center', color='white', fontsize=8)

# 3.2 解码器位置对编码器输出的关注（第一行）
ax = axes[0, 1]
q_words = ['I', 'love', 'learning', 'NLP']
kv_words = ['我', '爱', '学习', '自然', '语言', '处理']
x = np.arange(len(kv_words))
width = 0.2

for i in range(min(4, seq_len_q)):
    bars = ax.bar(x + i*width, weights[i], width, label=f'Decoder pos {i+1}: {q_words[i]}', 
                  alpha=0.7)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Encoder positions', fontsize=10)
ax.set_ylabel('Attention weight', fontsize=10)
ax.set_title('Decoder positions attending to encoder output', fontsize=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(kv_words, rotation=30, ha='right', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.1)

# 3.3 交叉注意力 vs 自注意力对比
ax = axes[1, 0]
categories = ['Self-Attention', 'Cross-Attention']
input_types = ['Same sequence', 'Different sequences']
colors = ['steelblue', 'coral']

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, [1, 0], width, label='Same sequence', color='steelblue', alpha=0.7)
bars2 = ax.bar(x + width/2, [0, 1], width, label='Different sequences', color='coral', alpha=0.7)

ax.set_ylabel('Applies to', fontsize=10)
ax.set_title('Self-Attention vs Cross-Attention', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.5)

for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
            'Yes' if bar1.get_height() > 0 else 'No', ha='center', va='bottom', fontsize=9)
    ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
            'Yes' if bar2.get_height() > 0 else 'No', ha='center', va='bottom', fontsize=9)

# 3.4 交叉注意力流程图
ax = axes[1, 1]
ax.text(0.5, 0.95, 'Decoder Input', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.75, 'Query (Q)', fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.55, 'Cross-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.35, 'Key (K) & Value (V)', fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.15, 'Encoder Output', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.arrow(0.5, 0.90, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.70, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.50, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.30, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.10, 0, 0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.axis('off')
ax.set_title('Cross-Attention Flow', fontsize=12)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'cross_attention.png'))
print("Visualization saved to 'images/cross_attention.png'")

# 4. 交叉注意力特点
print("\n4. Cross-Attention Features...")

print("""
Cross-Attention Features:
1. Cross-Sequence Attention
   - Query from decoder
   - Key and Value from encoder
   - Captures alignment between sequences

2. Information Transfer
   - Encoder output transferred to decoder
   - Enables encoder-decoder interaction
   - Dynamic context fusion

3. Multiple Heads
   - Captures different representation subspaces
   - Improves model capacity
   - Better alignment capture

4. Applications
   - Machine Translation
   - Text Summarization
   - Question Answering
   - Dialogue Systems
""")

# 5. 交叉注意力与自注意力对比
print("\n5. Cross-Attention vs Self-Attention...")

print("""
| Feature | Self-Attention | Cross-Attention |
|---------|----------------|-----------------|
| Input | Same sequence | Two sequences |
| Query | Same sequence | Decoder |
| Key | Same sequence | Encoder |
| Value | Same sequence | Encoder |
| Application | Encoder, Masked | Decoder |
| Purpose | Internal dependencies | Sequence alignment |
""")

# 6. 应用场景
print("\n6. Applications...")

print("""
Cross-Attention Applications:
1. Machine Translation:
   - Input: Source language sentence (encoder)
   - Output: Target language sentence (decoder)
   - Role: Decoder attends to different parts of source sentence
   - Example: Translating "I love learning" - decoder attends to positions

2. Text Summarization:
   - Input: Long text (encoder)
   - Output: Summary (decoder)
   - Role: Decoder attends to different parts of long text
   - Example: Generating summary - decoder attends to key sentences

3. Question Answering:
   - Input: Question + Context (encoder)
   - Output: Answer (decoder)
   - Role: Decoder attends to question and context
   - Example: Answering question - decoder attends to keywords and context

4. Dialogue Systems:
   - Input: Dialogue history (encoder)
   - Output: Response (decoder)
   - Role: Decoder attends to different parts of dialogue history
   - Example: Generating response - decoder attends to relevant parts

5. Image Captioning:
   - Input: Image features (encoder)
   - Output: Caption (decoder)
   - Role: Decoder attends to different regions of image
   - Example: Generating caption - decoder attends to image regions
""")

# 7. 总结
print("\n" + "=" * 70)
print("Cross-Attention Summary")
print("=" * 70)

print("""
Key Concepts:
1. Cross-Attention is the core component of Transformer architecture
2. Query from decoder, Key and Value from encoder
3. Captures alignment between input and output sequences
4. Enables information transfer between encoder and decoder
5. Widely used in Seq2Seq tasks

Cross-Attention Features:
- Cross-sequence attention
- Information transfer
- Multiple heads
- Dynamic context fusion

Cross-Attention vs Self-Attention:
- Self-Attention: Same sequence, internal dependencies
- Cross-Attention: Two sequences, sequence alignment
""")

print("=" * 70)
print("Cross-Attention Demo completed!")
print("=" * 70)
