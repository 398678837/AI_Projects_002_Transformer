"""
位置编码演示
Positional Encoding
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("位置编码演示")
print("=" * 70)

# 1. 位置编码概念
print("\n1. 位置编码概念...")

print("""
位置编码 (Positional Encoding):
- 为序列中的每个位置添加位置信息
- 让模型知道词的顺序
- 正弦余弦编码
""")

# 2. 位置编码计算
print("\n2. 位置编码计算...")

def get_positional_encoding(max_len, d_model):
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

max_len = 20
d_model = 16
pe = get_positional_encoding(max_len, d_model)

print(f"  位置编码形状: {pe.shape}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
im = ax.imshow(pe[:10].T, cmap='RdBu', aspect='auto')
ax.set_xlabel('位置')
ax.set_ylabel('维度')
ax.set_title('位置编码热力图')
plt.colorbar(im, ax=ax)

ax = axes[1]
ax.plot(pe[:20, 0], label='偶数维度 (sin)', linewidth=2)
ax.plot(pe[:20, 1], label='奇数维度 (cos)', linewidth=2)
ax.set_xlabel('位置')
ax.set_ylabel('编码值')
ax.set_title('位置编码曲线')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/positional_encoding.png')
print("可视化已保存为 'images/positional_encoding.png'")

# 4. 总结
print("\n" + "=" * 70)
print("位置编码总结")
print("=" * 70)
print("""
位置编码特点:
1. 每个位置有唯一编码
2. 不同位置有规律性
3. 可以推广到任意长度
""")
print("=" * 70)
print("\nPositional Encoding Demo完成！")
