"""
解码器演示
Decoder
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("解码器演示")
print("=" * 70)

# 1. 解码器概念
print("\n1. 解码器概念...")

print("""
解码器 (Decoder):
- 自回归生成输出序列
- 包含自注意力和交叉注意力
- 单向建模
""")

# 2. 解码器结构
print("\n2. 解码器结构...")

class SimpleDecoder:
    def __init__(self, d_model=8, num_heads=2):
        self.d_model = d_model
        self.num_heads = num_heads
    
    def forward(self, x, encoder_output):
        self_attention = np.random.randn(*x.shape)
        cross_attention = np.random.randn(*x.shape)
        ff_output = np.random.randn(*x.shape)
        output = self_attention + cross_attention + ff_output + x
        return output

decoder = SimpleDecoder()
input_seq = np.random.randn(5, 8)
encoder_out = np.random.randn(5, 8)
output = decoder.forward(input_seq, encoder_out)

print(f"  输入形状: {input_seq.shape}")
print(f"  编码器输出形状: {encoder_out.shape}")
print(f"  输出形状: {output.shape}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.text(0.5, 0.9, 'Decoder', fontsize=20, ha='center', transform=ax.transAxes)
ax.text(0.5, 0.7, 'Masked Self-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightblue'))
ax.text(0.5, 0.5, 'Cross-Attention', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(0.5, 0.3, 'Feed Forward', fontsize=12, ha='center', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.axis('off')
ax.set_title('解码器结构')

ax = axes[1]
steps = ['步1', '步2', '步3', '步4', '步5']
time = [1, 2, 3, 4, 5]
ax.bar(steps, time, color='coral', alpha=0.7)
ax.set_ylabel('生成时间步')
ax.set_title('自回归生成过程')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/decoder.png')
print("可视化已保存为 'images/decoder.png'")

# 4. 总结
print("\n" + "=" * 70)
print("解码器总结")
print("=" * 70)
print("""
解码器特点:
1. 自回归生成
2. 掩码自注意力
3. 交叉注意力
""")
print("=" * 70)
print("\nDecoder Demo完成！")
