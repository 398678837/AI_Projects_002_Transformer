"""
迁移学习演示
利用预训练模型
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("迁移学习演示")
print("=" * 70)

# 1. 迁移学习概念
print("\n1. 迁移学习概念...")

print("""
迁移学习 (Transfer Learning):
- 利用预训练模型的知识
- 减少训练数据和时间
- 提高模型性能

常见策略:
1. 特征提取: 冻结预训练层，训练新分类器
2. 微调: 解冻部分层联合训练
3. 完全训练: 解冻所有层从头训练
""")

# 2. 模拟特征提取
print("\n2. 模拟特征提取...")

class SimpleFeatureExtractor:
    def __init__(self):
        self.features = {
            'layer1': np.random.randn(64, 224, 224),
            'layer2': np.random.randn(128, 112, 112),
            'layer3': np.random.randn(256, 56, 56),
            'layer4': np.random.randn(512, 28, 28)
        }
    
    def extract(self, image):
        return np.random.randn(512)
    
    def visualize_features(self):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
        sizes = [64, 128, 256, 512]
        
        for i, (name, size) in enumerate(zip(layer_names, sizes)):
            fake_feature = np.random.randn(size, 4, 4)
            avg_feature = np.mean(fake_feature, axis=0)
            
            axes[i].imshow(avg_feature, cmap='viridis')
            axes[i].set_title(f'{name}\n{size} filters')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('images/transfer_features.png')
        print("可视化已保存为 'images/transfer_features.png'")

extractor = SimpleFeatureExtractor()
extractor.visualize_features()

# 3. 对比实验
print("\n3. 对比不同策略...")

strategies = ['从零训练', '特征提取', '微调', '完全微调']
data_needed = [100000, 1000, 5000, 10000]
accuracy = [0.72, 0.85, 0.92, 0.94]
time_cost = [24, 1, 4, 12]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.bar(strategies, data_needed, color='steelblue', alpha=0.7)
ax.set_ylabel('所需数据量')
ax.set_title('数据需求对比')
ax.set_xticklabels(strategies, rotation=15)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(strategies, accuracy, color='coral', alpha=0.7)
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy对比')
ax.set_xticklabels(strategies, rotation=15)
ax.set_ylim(0.5, 1.0)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.bar(strategies, time_cost, color='green', alpha=0.7)
ax.set_ylabel('训练时间 (小时)')
ax.set_title('训练时间对比')
ax.set_xticklabels(strategies, rotation=15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/transfer_learning.png')
print("可视化已保存为 'images/transfer_learning.png'")

# 4. 总结
print("\n" + "=" * 70)
print("迁移学习总结")
print("=" * 70)
print("""
| 策略 | 数据需求 | Accuracy | 训练时间 |
|------|----------|--------|----------|
| 从零训练 | 高 | 低 | 长 |
| 特征提取 | 低 | 中等 | 短 |
| 微调 | 中等 | 高 | 中 |
| 完全微调 | 中等 | 最高 | 长 |

适用场景:
- 数据少: 特征提取
- 数据中等: 微调
- 数据多: 完全训练
""")
print("=" * 70)
print("\nTransfer Learning Demo完成！")
