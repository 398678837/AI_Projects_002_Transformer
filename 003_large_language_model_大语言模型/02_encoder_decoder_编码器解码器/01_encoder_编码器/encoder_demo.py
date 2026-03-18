"""
编码器演示
Encoder Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Encoder Demo")
print("=" * 70)

# 1. 编码器概念
print("\n1. Encoder Concept...")

print("""
Encoder (编码器):
- 将输入序列编码为表示
- 包含多头自注意力层和前馈网络
- 双向建模上下文
- 残差连接和层归一化
""")

# 2. 编码器结构
print("\n2. Encoder Structure...")

class EncoderLayer:
    """编码器层"""
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
    
    def self_attention(self, x):
        """自注意力计算"""
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
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
    
    def forward(self, x):
        """前向传播"""
        # 多头自注意力
        attention_output, attention_weights = self.self_attention(x)
        
        # 残差连接和层归一化
        x = self.layer_norm(x + attention_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        output = self.layer_norm(x + ff_output)
        
        return output, attention_weights

class Encoder:
    """编码器"""
    def __init__(self, d_model=8, num_heads=2, d_ff=16, num_layers=2):
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x):
        """前向传播"""
        attention_weights_list = []
        
        for i, layer in enumerate(self.layers):
            x, attention_weights = layer.forward(x)
            attention_weights_list.append(attention_weights)
        
        return x, attention_weights_list

# 创建编码器
encoder = Encoder(d_model=8, num_heads=2, d_ff=16, num_layers=2)

# 输入序列（5个词，每个词8维）
input_seq = np.random.randn(5, 8)

# 前向传播
output, attention_weights_list = encoder.forward(input_seq)

print(f"  输入序列形状: {input_seq.shape}")
print(f"  输出序列形状: {output.shape}")
print(f"  编码器层数: {encoder.num_layers}")
print(f"  注意力头数: {encoder.layers[0].num_heads}")

# 3. 可视化
print("\n3. Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 编码器结构
ax = axes[0, 0]
layers = ['Input', 'Multi-Head Attention', 'Feed Forward', 'Output']
dims = [8, 8, 16, 8]
ax.bar(layers, dims, color='steelblue', alpha=0.7)
ax.set_ylabel('Dimension')
ax.set_title('Encoder Layer Dimensions')
ax.grid(True, alpha=0.3)

# 3.2 编码器堆叠
ax = axes[0, 1]
layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
heights = [1] * 6
colors = ['steelblue', 'lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'blue']
ax.bar(layer_names[:len(encoder.layers)], heights[:len(encoder.layers)], color=colors[:len(encoder.layers)])
ax.set_ylabel('Stack')
ax.set_title('Encoder Stacking')
ax.set_ylim(0, 1.5)
for i, v in enumerate(heights[:len(encoder.layers)]):
    ax.text(i, v + 0.1, str(i+1), ha='center', va='bottom', fontsize=12)

# 3.3 注意力权重可视化（第一层）
ax = axes[1, 0]
attention_weights = attention_weights_list[0]
im = ax.imshow(attention_weights, cmap='viridis')
ax.set_title('Attention Weights (Layer 1)')
ax.set_xlabel('Key positions')
ax.set_ylabel('Query positions')
plt.colorbar(im, ax=ax)

# 添加数值标签
for i in range(attention_weights.shape[0]):
    for j in range(attention_weights.shape[1]):
        ax.text(j, i, f'{attention_weights[i, j]:.2f}', 
                ha='center', va='center', color='white', fontsize=8)

# 3.4 编码器流程图
ax = axes[1, 1]
ax.text(0.5, 0.9, 'Input', fontsize=14, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(0.5, 0.7, 'Multi-Head\nSelf-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.5, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.3, 'Feed Forward\nNetwork', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.1, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.arrow(0.5, 0.85, 0, -0.25, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.65, 0, -0.25, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.45, 0, -0.25, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.arrow(0.5, 0.25, 0, -0.25, transform=ax.transAxes, color='black', head_width=0.02, head_length=0.03)
ax.axis('off')
ax.set_title('Encoder Layer Flow')

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'encoder.png'))
print("Visualization saved to 'images/encoder.png'")

# 4. 编码器特点
print("\n4. Encoder Features...")

print("""
Encoder Features:
1. Bidirectional Self-Attention
   - Each position can attend to all positions
   - Captures context from both directions
   
2. Residual Connections
   - Output = LayerNorm(X + Sublayer(X))
   - Relieves gradient vanishing problem
   
3. Layer Normalization
   - Stabilizes training process
   - Accelerates convergence
   
4. Parallel Computation
   - Can be computed in parallel
   - Much faster than RNN
""")

# 5. 编码器与解码器对比
print("\n5. Encoder vs Decoder...")

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
Encoder Applications:
1. NLP:
   - BERT: Pre-trained model based on encoder
   - Text Classification: Captures context information
   - Named Entity Recognition: Identifies entities in text
   - Question Answering: Understands question and context

2. Computer Vision:
   - Vision Transformer: Applies encoder to images
   - Image Classification: Captures global information
   - Object Detection: Understands object relationships

3. Audio Processing:
   - Speech Recognition: Captures temporal information
   - Audio Classification: Identifies patterns in audio
""")

# 7. 总结
print("\n" + "=" * 70)
print("Encoder Summary")
print("=" * 70)

print("""
Key Concepts:
1. Encoder is the core component of Transformer architecture
2. Uses multi-head self-attention to capture context
3. Includes residual connections and layer normalization
4. Can be computed in parallel
5. Widely used in BERT and other pre-trained models

Encoder Features:
- Bidirectional modeling
- Parallel computation
- Residual connections
- Layer normalization

Encoder vs Decoder:
- Encoder: Bidirectional attention, no cross-attention
- Decoder: Masked attention, with cross-attention
""")

print("=" * 70)
print("Encoder Demo completed!")
print("=" * 70)
