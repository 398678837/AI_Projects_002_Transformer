"""
经典CNN架构演示
LeNet, AlexNet, VGG, ResNet
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("经典CNN架构演示")
print("=" * 70)

# 1. LeNet-5
print("\n1. LeNet-5 (1998)")
print("-" * 50)
print("""
结构:
- 输入: 32x32 灰度图
- C1: 6个5x5卷积核 -> 28x28
- S2: 6个2x2池化 -> 14x14
- C3: 16个5x5卷积核 -> 10x10
- S4: 16个2x2池化 -> 5x5
- C5: 120个5x5卷积
- F6: 84个全连接
- 输出: 10
""")

# 2. AlexNet
print("\n2. AlexNet (2012)")
print("-" * 50)
print("""
结构:
- 输入: 224x224 RGB图
- 5个卷积层
- 3个全连接层
- 参数: 60M
- 特点: ReLU、Dropout、GPU训练
""")

# 3. VGG
print("\n3. VGG (2014)")
print("-" * 50)
print("""
结构:
- VGG16: 16层
- VGG19: 19层
- 特点: 3x3卷积堆叠
- 参数: 138M
""")

# 4. ResNet
print("\n4. ResNet (2015)")
print("-" * 50)
print("""
结构:
- 残差块: y = F(x) + x
- 解决梯度消失问题
- 可训练超过1000层
- 经典版本: ResNet50, ResNet101
""")

# 5. 可视化架构对比
print("\n5. 架构对比可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
architectures = ['LeNet-5', 'AlexNet', 'VGG-16', 'VGG-19', 'ResNet-50']
params = [0.06, 61, 138, 144, 25.5]
layers = [7, 8, 16, 19, 50]

x = np.arange(len(architectures))
width = 0.35

bars1 = ax.bar(x - width/2, [p/1000 for p in params], width, label='参数 (M)', color='steelblue', alpha=0.7)
ax.set_ylabel('参数 (百万)')
ax.set_title('参数量对比')
ax.set_xticks(x)
ax.set_xticklabels(architectures, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3)

bars2 = ax.bar(x + width/2, layers, width, label='层数', color='coral', alpha=0.7)
ax.set_ylabel('层数')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
performance = {
    'LeNet-5': 99.2,
    'AlexNet': 63.3,
    'VGG-16': 68.5,
    'ResNet-50': 69.2
}
ax.bar(performance.keys(), performance.values(), color=['steelblue', 'coral', 'green', 'purple'], alpha=0.7)
ax.set_ylabel('Top-5 错误率 (%)')
ax.set_title('ImageNet性能对比')
ax.axhline(y=5, color='red', linestyle='--', label='人类水平')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('images/cnn_architectures.png')
print("可视化已保存为 'images/cnn_architectures.png'")

# 6. 残差块图示
fig, ax = plt.subplots(figsize=(8, 6))

ax.add_patch(plt.Rectangle((1, 2), 3, 2, facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(2.5, 3, 'Conv', ha='center', va='center', fontsize=12)
ax.text(2.5, 4.2, 'F(x)', ha='center', va='center', fontsize=12, color='blue')

ax.add_patch(plt.Rectangle((1, -1), 3, 2, facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(2.5, 0, 'Conv', ha='center', va='center', fontsize=12)
ax.text(2.5, -0.2, 'BN+ReLU', ha='center', va='center', fontsize=8)

ax.add_patch(plt.Rectangle((1, -4), 3, 2, facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(2.5, -3, 'Conv', ha='center', va='center', fontsize=12)

ax.annotate('', xy=(4.5, 3), xytext=(4.5, -2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(5, 0.5, '+', fontsize=20, color='red')

ax.text(4.5, 4, 'x', fontsize=14, color='green')
ax.annotate('', xy=(4.5, 3), xytext=(4.5, 4),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

ax.add_patch(plt.Rectangle((1, -5.5), 3, 1.5, facecolor='lightgreen', edgecolor='black', linewidth=2))
ax.text(2.5, -4.75, 'ReLU', ha='center', va='center', fontsize=12)

ax.set_xlim(0, 7)
ax.set_ylim(-7, 6)
ax.set_title('ResNet残差块结构', fontsize=14)
ax.axis('off')

plt.tight_layout()
plt.savefig('images/resnet_block.png')
print("可视化已保存为 'images/resnet_block.png'")

# 7. 总结
print("\n" + "=" * 70)
print("经典CNN架构总结")
print("=" * 70)
print("""
| 架构 | 年份 | 层数 | 参数 | 特点 |
|------|------|------|------|------|
| LeNet-5 | 1998 | 7 | 60K | 早期经典 |
| AlexNet | 2012 | 8 | 61M | 深度学习复兴 |
| VGG | 2014 | 16/19 | 138M | 结构简单 |
| ResNet | 2015 | 50+ | 25M+ | 残差连接 |

ResNet的残差连接是最重要的创新。
""")
print("=" * 70)
print("\nClassic CNN Architectures Demo完成！")
