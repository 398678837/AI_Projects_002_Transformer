"""
t-SNE（t分布邻域嵌入）降维算法的scikit-learn实现
使用MNIST数据集的子集进行降维和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import time

# 1. 加载MNIST数据集（子集）
print("加载MNIST数据集（子集）...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# 使用子集（1000个样本）以加快速度
np.random.seed(42)
indices = np.random.choice(len(X), 1000, replace=False)
X_subset = X[indices]
y_subset = y[indices].astype(int)

print(f"数据集形状: X={X_subset.shape}, y={y_subset.shape}")
print(f"Class: {np.unique(y_subset)}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# 3. 应用t-SNE降维
print("\n应用t-SNE降维...")
start_time = time.time()

# 创建t-SNE模型
tsne = TSNE(
    n_components=2,  # 降维到2维
    perplexity=30,   # 困惑度，通常设为5-50
    learning_rate=200,  # 学习率
    n_iter=1000,    # 最大迭代次数
    random_state=42,
    verbose=1
)

# 训练模型并降维
X_tsne = tsne.fit_transform(X_scaled)

end_time = time.time()
print(f"t-SNE降维完成，耗时: {end_time - start_time:.2f}秒")
print(f"降维后数据形状: {X_tsne.shape}")

# 4. 可视化降维结果
print("\n可视化降维结果...")

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='数字Class')
plt.title('t-SNE Dimensionality Reduction - MNIST Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/tsne_mnist_visualization.png')
print("t-SNE可视化已保存为 'images/tsne_mnist_visualization.png'")

# 5. 尝试不同的困惑度
print("\n尝试不同的困惑度...")
perplexities = [5, 15, 30, 50]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, perplexity in enumerate(perplexities):
    print(f"训练困惑度={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1000,
        random_state=42,
        verbose=0
    )
    X_tsne_perp = tsne.fit_transform(X_scaled)
    
    ax = axes[idx]
    scatter = ax.scatter(X_tsne_perp[:, 0], X_tsne_perp[:, 1], c=y_subset, cmap='tab10', s=30, alpha=0.7)
    ax.set_title(f't-SNE 困惑度={perplexity}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/tsne_perplexity_comparison.png')
print("不同困惑度对比已保存为 'images/tsne_perplexity_comparison.png'")

# 6. 分析t-SNE的参数
print("\nt-SNE参数说明:")
print("- perplexity: 困惑度，控制局部和全局结构的平衡")
print("  - 较小值：关注局部结构")
print("  - 较大值：关注全局结构")
print("  - 推荐范围：5-50")
print()
print("- learning_rate: 学习率")
print("  - 通常设置为100-1000")
print("  - 如果结果看起来像一个球，可能学习率太高")
print("  - 如果结果看起来太分散，可能学习率太低")
print()
print("- n_iter: 迭代次数")
print("  - 至少250次，通常1000次足够")
print("  - K-L散度应该在迭代过程中稳定")

print("\nscikit-learn t-SNE Demo完成！")
