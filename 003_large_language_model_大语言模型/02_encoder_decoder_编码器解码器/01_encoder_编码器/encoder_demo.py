"""
编码器演示
Encoder
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("编码器演示")
print("=" * 70)

# 1. 编码器概念
print("\n1. 编码器概念...")

print("""
编码器 (Encoder):
- 将输入序列编码为表示
- 包含自注意力层和前馈网络
- 双向建模上下文
""")

# 2. 编码器结构
print("\n2. 编码器结构...")

class SimpleEncoder:
    def __init__(self, d_model=8, num_heads=2, d_ff=16):
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, x):
        attention_output = np.random.randn(*x.shape)
        ff_output = np.random.randn(*x.shape)
        output = attention_output + ff_output + x
        return output

encoder = SimpleEncoder()
input_seq = np.random.randn(5, 8)
output = encoder.forward(input_seq)

print(f"  输入形状: {input_seq.shape}")
print(f"  输出形状: {output.shape}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
layers = ['输入', '自注意力', '前馈网络', '输出']
dims = [8, 8, 16, 8]
ax.bar(layers, dims, color='steelblue', alpha=0.7)
ax.set_ylabel('维度')
ax.set_title('编码器各层维度')
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.text(0.5, 0.8, 'Encoder', fontsize=20, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.6, 'Multi-Head Self-Attention', fontsize=12, ha='center', transform=ax.transAxes, 
       bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.4, 'Feed Forward Network', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(0.5, 0.2, 'Add & Norm', fontsize=12, ha='center', transform=ax.transAxes)
ax.axis('off')
ax.set_title('编码器结构')

plt.tight_layout()
plt.savefig('images/encoder.png')
print("可视化已保存为 'images/encoder.png'")

# 4. 总结
print("\n" + "=" * 70)
print("编码器总结")
print("=" * 70)
print("""
编码器特点:
1. 双向自注意力
2. 包含残差连接和层归一化
3. 输出上下文表示
""")
print("=" * 70)
print("\nEncoder Demo完成！")
