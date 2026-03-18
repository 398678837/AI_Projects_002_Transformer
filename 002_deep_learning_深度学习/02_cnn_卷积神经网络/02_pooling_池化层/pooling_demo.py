"""
池化层演示
最大池化和平均池化
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("池化层演示")
print("=" * 70)

# 1. 池化操作
print("\n1. 池化操作定义...")

def max_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    result = np.zeros((out_h, out_w))
    
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            region = image[i:i+pool_size, j:j+pool_size]
            result[i//stride, j//stride] = np.max(region)
    
    return result

def avg_pooling(image, pool_size=2, stride=2):
    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    
    result = np.zeros((out_h, out_w))
    
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            region = image[i:i+pool_size, j:j+pool_size]
            result[i//stride, j//stride] = np.mean(region)
    
    return result

# 2. 示例
print("\n2. 池化示例...")

image = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=float)

max_pool = max_pooling(image, pool_size=2, stride=2)
avg_pool = avg_pooling(image, pool_size=2, stride=2)

print(f"  原始图像:\n{image}")
print(f"\n  最大池化:\n{max_pool}")
print(f"\n  平均池化:\n{avg_pool}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

ax = axes[0]
im = ax.imshow(image, cmap='viridis')
ax.set_title('原始特征图 (4x4)')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{int(image[i,j])}', ha='center', va='center', color='white')
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(max_pool, cmap='viridis')
ax.set_title('最大池化 (2x2)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{int(max_pool[i,j])}', ha='center', va='center', color='white')
plt.colorbar(im, ax=ax)

ax = axes[2]
im = ax.imshow(avg_pool, cmap='viridis')
ax.set_title('平均池化 (2x2)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{int(avg_pool[i,j])}', ha='center', va='center', color='white')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'pooling.png'))
print("可视化已保存为 'images/pooling.png'")

# 4. 总结
print("\n" + "=" * 70)
print("池化层总结")
print("=" * 70)
print("""
| 类型 | 作用 | 优点 |
|------|------|------|
| 最大池化 | 提取最显著特征 | 保持特征 |
| 平均池化 | 平滑特征 | 减少噪声 |

作用:
1. 减少空间尺寸
2. 减少参数数量
3. 防止过拟合
4. 提供平移不变性
""")
print("=" * 70)
print("\nPooling Demo完成！")
