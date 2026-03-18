"""
卷积层演示
卷积神经网络基础
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("卷积层演示")
print("=" * 70)

# 1. 卷积操作
print("\n1. 卷积操作定义...")

def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    
    h, w = image.shape
    kh, kw = kernel.shape
    
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    
    result = np.zeros((out_h, out_w))
    
    for i in range(0, out_h * stride, stride):
        for j in range(0, out_w * stride, stride):
            region = image[i:i+kh, j:j+kw]
            result[i//stride, j//stride] = np.sum(region * kernel)
    
    return result

# 2. 创建示例图像和卷积核
print("\n2. 创建示例...")

image = np.zeros((10, 10))
image[3:7, 3:7] = 1

sobel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, 1]])

# 3. 应用卷积
print("\n3. 应用卷积核...")

edges_h = conv2d(image, sobel_h, stride=1, padding=1)
edges_v = conv2d(image, sobel_v, stride=1, padding=1)
edges = conv2d(image, edge_kernel, stride=1, padding=1)

print(f"  原始图像形状: {image.shape}")
print(f"  卷积输出形状: {edges.shape}")

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

ax = axes[0]
ax.imshow(image, cmap='gray')
ax.set_title('原始图像')
ax.axis('off')

ax = axes[1]
ax.imshow(edges_h, cmap='gray')
ax.set_title('水平边缘检测')
ax.axis('off')

ax = axes[2]
ax.imshow(edges_v, cmap='gray')
ax.set_title('垂直边缘检测')
ax.axis('off')

ax = axes[3]
ax.imshow(edges, cmap='gray')
ax.set_title('边缘检测')
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'convolution.png'))
print("可视化已保存为 'images/convolution.png'")

# 5. CNN结构图
print("\n5. CNN结构...")

fig, ax = plt.subplots(figsize=(10, 6))

layers = ['Input层', '卷积层1', '池化层1', '卷积层2', '池化层2', '全连接层', '输出层']
sizes = [224, 224, 112, 112, 56, 56, 10]
colors = ['steelblue', 'coral', 'lightgreen', 'coral', 'lightgreen', 'coral', 'steelblue']

for i, (layer, size, color) in enumerate(zip(layers, sizes, colors)):
    y = 10 - i * 2
    rect = plt.Rectangle((10, y-1), size/10, 1.5, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(12, y-0.2, f'{layer}\n{size}x{size}', fontsize=10, va='top')

ax.set_xlim(0, 40)
ax.set_ylim(0, 12)
ax.set_title('CNN网络结构', fontsize=14)
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'cnn_structure.png'))
print("可视化已保存为 'images/cnn_structure.png'")

# 6. 总结
print("\n" + "=" * 70)
print("卷积层总结")
print("=" * 70)
print("""
关键概念:

1. 卷积核 (Kernel):
   - 小矩阵 (如3x3, 5x5)
   - 提取特定特征

2. 步长 (Stride):
   - 卷积核移动的步长
   - 影响输出尺寸

3. 填充 (Padding):
   - 保持空间尺寸
   - 防止边缘信息丢失

4. 特征图 (Feature Map):
   - 卷积输出
   - 表示提取的特征
""")
print("=" * 70)
print("\nConvolution Demo完成！")
