"""
解码器演示
Decoder Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Decoder Demo")
print("=" * 70)

# 1. 解码器概念
print("\n1. Decoder Concept...")

print("""
Decoder (解码器):
- 自回归生成输出序列
- 包含掩码自注意力和交叉注意力
- 单向建模
""")

# 2. 解码器结构
print("\n2. Decoder Structure...")

class DecoderLayer:
    """解码器层"""
    def __init__(self, d_model=8, num_heads=2, d_ff=16):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # 初始化权重
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)
        self.W_1 = np.random.randn(d_model, d_ff)
        self.W_2 = np.random.randn(d_ff, d_model)
    
    def softmax(self, x):
        """softmax函数"""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def generate_mask(self, size):
        """生成上三角掩码"""
        mask = np.triu(np.ones((size, size)), k=1)
        return mask
    
    def masked_self_attention(self, x):
        """掩码自注意力计算"""
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        d_k = Q.shape[-1]
        attention_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        
        # 生成掩码
        mask = self.generate_mask(x.shape[0])
        attention_scores = np.where(mask == 1, -np.inf, attention_scores)
        
        attention_weights = self.softmax(attention_scores)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def cross_attention(self, x, encoder_output):
        """交叉注意力计算"""
        Q = np.matmul(x, self.W_q)
        K = np.matmul(encoder_output, self.W_k)
        V = np.matmul(encoder_output, self.W_v)
        
        d_k = Q.shape[-1]
        attention_scores = np.matmul(Q, K.T) / np.sqrt(d_k)
        attention_weights = self.softmax(attention_scores)
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def feed_forward(self, x):
        """前馈网络"""
        hidden = np.maximum(0, np.matmul(x, self.W_1))  # ReLU
        output = np.matmul(hidden, self.W_2)
        return output
    
    def layer_norm(self, x, eps=1e-6):
        """层归一化"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def forward(self, x, encoder_output):
        """前向传播"""
        # 掩码多头自注意力
        self_attention_output, self_attention_weights = self.masked_self_attention(x)
        
        # 残差连接和层归一化
        x = self.layer_norm(x + self_attention_output)
        
        # 交叉注意力
        cross_attention_output, cross_attention_weights = self.cross_attention(x, encoder_output)
        
        # 残差连接和层归一化
        x = self.layer_norm(x + cross_attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        output = self.layer_norm(x + ff_output)
        
        return output, self_attention_weights, cross_attention_weights

class Decoder:
    """解码器"""
    def __init__(self, d_model=8, num_heads=2, d_ff=16, num_layers=2):
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x, encoder_output):
        """前向传播"""
        self_attention_weights_list = []
        cross_attention_weights_list = []
        
        for i, layer in enumerate(self.layers):
            x, self_attention_weights, cross_attention_weights = layer.forward(x, encoder_output)
            self_attention_weights_list.append(self_attention_weights)
            cross_attention_weights_list.append(cross_attention_weights)
        
        return x, self_attention_weights_list, cross_attention_weights_list

# 创建解码器
decoder = Decoder(d_model=8, num_heads=2, d_ff=16, num_layers=2)

# 输入序列（5个词，每个词8维）
input_seq = np.random.randn(5, 8)

# 编码器输出（5个词，每个词8维）
encoder_output = np.random.randn(5, 8)

# 前向传播
output, self_attention_weights_list, cross_attention_weights_list = decoder.forward(input_seq, encoder_output)

print(f"  输入序列形状: {input_seq.shape}")
print(f"  编码器输出形状: {encoder_output.shape}")
print(f"  输出序列形状: {output.shape}")
print(f"  解码器层数: {decoder.num_layers}")
print(f"  注意力头数: {decoder.layers[0].num_heads}")

# 3. 可视化
print("\n3. Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 解码器结构
ax = axes[0, 0]
layers = ['Input', 'Masked Self-Attention', 'Cross-Attention', 'Feed Forward', 'Output']
dims = [8, 8, 8, 16, 8]
ax.bar(layers, dims, color='coral', alpha=0.7)
ax.set_ylabel('Dimension')
ax.set_title('Decoder Layer Dimensions')
ax.grid(True, alpha=0.3)

# 3.2 解码器堆叠
ax = axes[0, 1]
layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
heights = [1] * 6
colors = ['coral', 'lightcoral', 'darksalmon', 'salmon', 'lightsalmon', 'tomato']
ax.bar(layer_names[:len(decoder.layers)], heights[:len(decoder.layers)], color=colors[:len(decoder.layers)])
ax.set_ylabel('Stack')
ax.set_title('Decoder Stacking')
ax.set_ylim(0, 1.5)
for i, v in enumerate(heights[:len(decoder.layers)]):
    ax.text(i, v + 0.1, str(i+1), ha='center', va='bottom', fontsize=12)

# 3.3 掩码自注意力权重可视化（第一层）
ax = axes[1, 0]
self_attention_weights = self_attention_weights_list[0]
im = ax.imshow(self_attention_weights, cmap='viridis')
ax.set_title('Masked Self-Attention Weights (Layer 1)')
ax.set_xlabel('Key positions')
ax.set_ylabel('Query positions')
plt.colorbar(im, ax=ax)

# 添加数值标签
for i in range(self_attention_weights.shape[0]):
    for j in range(self_attention_weights.shape[1]):
        ax.text(j, i, f'{self_attention_weights[i, j]:.2f}', 
                ha='center', va='center', color='white', fontsize=8)

# 3.4 解码器流程图
ax = axes[1, 1]
ax.text(0.5, 0.95, 'Input', fontsize=14, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(0.5, 0.80, 'Masked Self-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.65, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.50, 'Cross-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.35, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.20, 'Feed Forward\nNetwork', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightpink'))
ax.text(0.5, 0.05, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.arrow(0.5, 0.90, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.75, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.60, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.45, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.30, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.15, 0, -0.15, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.axis('off')
ax.set_title('Decoder Layer Flow')

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'decoder.png'))
print("Visualization saved to 'images/decoder.png'")

# 4. 解码器特点
print("\n4. Decoder Features...")

print("""
Decoder Features:
1. Masked Self-Attention
   - Only attends to current and previous positions
   - Prevents information leakage
   - Ensures autoregressive generation
   
2. Cross-Attention
   - Attends to encoder output
   - Connects encoder and decoder
   - Transfers information
   
3. Autoregressive Generation
   - Generates output position by position
   - Each position depends on previous positions
   - Implements autoregressive generation
   
4. Residual Connections and Layer Normalization
   - Each sublayer has residual connection and layer normalization
   - Relieves gradient vanishing problem
   - Stabilizes training process
""")

# 5. 解码器与编码器对比
print("\n5. Decoder vs Encoder...")

print("""
| Feature | Encoder | Decoder |
|---------|---------|---------|
| Self-Attention | Bidirectional | Masked (Unidirectional) |
| Cross-Attention | None | Yes (with encoder output) |
| Output | Context Representation | Sequence Generation |
| Application | BERT | GPT, Machine Translation |
""")

# 6. 应用场景
print("\n6. Applications...")

print("""
Decoder Applications:
1. NLP:
   - GPT: Pre-trained generation model based on decoder
   - Machine Translation: Generates target language sequence
   - Text Summarization: Generates summary sequence
   - Dialogue System: Generates response sequence

2. Audio Processing:
   - Speech Synthesis: Generates speech sequence
   - Audio Generation: Generates audio sequence

3. Image Generation:
   - Image Captioning: Generates image description
   - Image Generation: Generates image sequence
""")

# 7. 总结
print("\n" + "=" * 70)
print("Decoder Summary")
print("=" * 70)

print("""
Key Concepts:
1. Decoder is the core component of Transformer architecture
2. Uses masked self-attention for autoregressive generation
3. Includes cross-attention to connect with encoder
4. Can be computed in parallel (except for generation)
5. Widely used in GPT and other generation models

Decoder Features:
- Masked self-attention
- Cross-attention
- Autoregressive generation
- Residual connections and layer normalization

Decoder vs Encoder:
- Encoder: Bidirectional attention, no cross-attention
- Decoder: Masked attention, with cross-attention
""")

print("=" * 70)
print("Decoder Demo completed!")
print("=" * 70)
