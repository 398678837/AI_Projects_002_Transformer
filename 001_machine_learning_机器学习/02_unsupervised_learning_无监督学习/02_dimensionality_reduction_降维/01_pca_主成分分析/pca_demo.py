"""
主成分分析（PCA）降维算法的scikit-learn实现
使用鸢尾花数据集进行降维分析
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 真实标签

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"Class名称: {iris.target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA降维
print("\n执行PCA降维...")

# 降维到2维
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# 降维到3维
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print(f"原始数据维度: {X_scaled.shape}")
print(f"PCA降维到2维后: {X_pca_2d.shape}")
print(f"PCA降维到3维后: {X_pca_3d.shape}")

# 4. 解释方差分析
print("\n解释方差分析:")

# 计算解释方差比率
explained_variance_ratio_2d = pca_2d.explained_variance_ratio_
explained_variance_ratio_3d = pca_3d.explained_variance_ratio_

print(f"2维PCA解释方差比率: {explained_variance_ratio_2d}")
print(f"2维PCA累计解释方差: {np.sum(explained_variance_ratio_2d):.4f}")
print(f"3维PCA解释方差比率: {explained_variance_ratio_3d}")
print(f"3维PCA累计解释方差: {np.sum(explained_variance_ratio_3d):.4f}")

# 5. 确定最佳降维维度
print("\n确定最佳降维维度...")

# 计算不同维度的解释方差
pca_full = PCA(n_components=4, random_state=42)
pca_full.fit(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance)

print("各主成分的解释方差比率:")
for i, ratio in enumerate(explained_variance):
    print(f"主成分 {i+1}: {ratio:.4f}")

print("\n累计解释方差:")
for i, cum_ratio in enumerate(cumulative_explained_variance):
    print(f"前 {i+1} 个主成分: {cum_ratio:.4f}")

# 6. 主成分分析
print("\n主成分分析:")

# 获取主成分载荷
components = pca_2d.components_
print("2维PCA的主成分载荷:")
for i, component in enumerate(components):
    print(f"主成分 {i+1}: {component}")

# 分析主成分与原始特征的关系
print("\n主成分与原始特征的关系:")
feature_names = iris.feature_names
for i, component in enumerate(components):
    print(f"主成分 {i+1} 与特征的相关性:")
    for j, feature in enumerate(feature_names):
        print(f"  {feature}: {component[j]:.4f}")

# 7. 可视化降维结果
print("\n可视化降维结果...")

# 2维PCA可视化
plt.figure(figsize=(12, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', s=50)
plt.title('PCA Dimensionality Reduction to 2D Results')
plt.xlabel(f'主成分 1 ({explained_variance_ratio_2d[0]:.2%} 方差)')
plt.ylabel(f'主成分 2 ({explained_variance_ratio_2d[1]:.2%} 方差)')
plt.colorbar(ticks=[0, 1, 2], label='Class')
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('images/pca_2d_visualization.png')
print("2维PCA可视化已保存为 'pca_2d_visualization.png'")

# 解释方差可视化
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.6, label='单个主成分的解释方差')
plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, 'r-', label='累计解释方差')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance Analysis')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('images/pca_explained_variance.png')
print("解释方差分析可视化已保存为 'pca_explained_variance.png'")

# 8. PCA在Cluster中的应用
print("\nPCA在Cluster中的应用...")

# 使用原始数据进行K-MeansCluster
kmeans_original = KMeans(n_clusters=3, random_state=42)
y_pred_original = kmeans_original.fit_predict(X_scaled)
silhouette_original = silhouette_score(X_scaled, y_pred_original)
print(f"原始数据K-MeansClusterSilhouette Score: {silhouette_original:.4f}")

# 使用PCA降维后的数据进行K-MeansCluster
kmeans_pca = KMeans(n_clusters=3, random_state=42)
y_pred_pca = kmeans_pca.fit_predict(X_pca_2d)
silhouette_pca = silhouette_score(X_pca_2d, y_pred_pca)
print(f"PCA降维后K-MeansClusterSilhouette Score: {silhouette_pca:.4f}")

# 可视化Clustering Results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='viridis', s=50)
plt.title('Original Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(ticks=[0, 1, 2], label='Class')

plt.subplot(1, 2, 2)
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_pred_pca, cmap='viridis', s=50)
plt.title('K-Means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(ticks=[0, 1, 2], label='Cluster')

plt.tight_layout()
plt.savefig('images/pca_clustering.png')
print("PCAClustering Results可视化已保存为 'pca_clustering.png'")

# 9. 特征提取
print("\n基于PCA的特征提取...")

# 使用前2个主成分作为新特征
X_new = X_pca_2d
print(f"提取的新特征形状: {X_new.shape}")

# 10. 重构原始数据
print("\n重构原始数据...")

# 使用PCA重构原始数据
X_reconstructed = pca_2d.inverse_transform(X_pca_2d)

# 计算重构误差
reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed))
print(f"重构误差: {reconstruction_error:.4f}")

print("\nscikit-learn PCA降维 Demo完成！")
