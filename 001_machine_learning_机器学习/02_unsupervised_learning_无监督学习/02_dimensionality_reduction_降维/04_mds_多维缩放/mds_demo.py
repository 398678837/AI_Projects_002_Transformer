"""
MDS（多维缩放）降维算法的scikit-learn实现
使用鸢尾花数据集进行降维和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"Class名称: {target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 计算距离矩阵
print("\n计算距离矩阵...")
dist_matrix = pairwise_distances(X_scaled, metric='euclidean')
print(f"距离矩阵形状: {dist_matrix.shape}")

# 4. 应用MDS降维
print("\n应用MDS降维...")

# 创建MDS模型（度量MDS）
mds = MDS(
    n_components=2,
    metric=True,
    n_init=4,
    max_iter=300,
    random_state=42,
    verbose=1
)

# 训练模型并降维
X_mds = mds.fit_transform(X_scaled)

print(f"降维后数据形状: {X_mds.shape}")
print(f"应力（Stress）: {mds.stress_:.4f}")

# 5. 可视化降维结果
print("\n可视化降维结果...")

plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_mds[y == i, 0], X_mds[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('MDS Dimensionality Reduction - Iris Dataset')
plt.xlabel('MDS 1')
plt.ylabel('MDS 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/mds_iris_visualization.png')
print("MDS可视化已保存为 'images/mds_iris_visualization.png'")

# 6. 对比度量MDS和非度量MDS
print("\n对比度量MDS和非度量MDS...")

# 非度量MDS
nmds = MDS(
    n_components=2,
    metric=False,
    n_init=4,
    max_iter=300,
    random_state=42,
    verbose=0
)
X_nmds = nmds.fit_transform(X_scaled)

print(f"非度量MDS应力: {nmds.stress_:.4f}")

# 绘制对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 度量MDS
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax1.scatter(X_mds[y == i, 0], X_mds[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
ax1.set_title(f'度量MDS (应力={mds.stress_:.4f})')
ax1.set_xlabel('MDS 1')
ax1.set_ylabel('MDS 2')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 非度量MDS
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax2.scatter(X_nmds[y == i, 0], X_nmds[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
ax2.set_title(f'非度量MDS (应力={nmds.stress_:.4f})')
ax2.set_xlabel('MDS 1')
ax2.set_ylabel('MDS 2')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/mds_metric_vs_nonmetric.png')
print("度量MDS与非度量MDS对比已保存为 'images/mds_metric_vs_nonmetric.png'")

# 7. MDS参数说明
print("\nMDS参数说明:")
print("- n_components: 降维后的维度")
print("- metric: 是否使用度量MDS")
print("  - True: 度量MDS，保持距离关系")
print("  - False: 非度量MDS，保持距离顺序")
print("- n_init: 初始化次数，取最好结果")
print("- max_iter: 最大迭代次数")
print("- dissimilarity: 相似度矩阵类型")
print("  - 'euclidean': 使用欧氏距离（默认）")
print("  - 'precomputed': 使用预计算的距离矩阵")

print("\n应力（Stress）:")
print("- 衡量降维后距离与原始距离的差异")
print("- 值越小越好")
print("- 应力<0.1: 优秀")
print("- 应力0.1-0.2: 良好")
print("- 应力>0.2: 一般")

print("\nscikit-learn MDS Demo完成！")
