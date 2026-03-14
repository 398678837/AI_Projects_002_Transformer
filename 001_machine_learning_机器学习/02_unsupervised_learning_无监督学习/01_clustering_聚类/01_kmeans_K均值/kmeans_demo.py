"""
K-MeansCluster算法的scikit-learn实现
使用鸢尾花数据集进行Cluster分析
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 真实标签（用于评估）

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"Class名称: {iris.target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-MeansCluster
print("\n执行K-MeansCluster...")
# 设置Cluster数量为3（与真实Class数一致）
k = 3

# 创建K-Means模型
kmeans = KMeans(n_clusters=k, random_state=42)

# 训练模型
kmeans.fit(X_scaled)

# 获取Clustering Results
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 4. 模型评估
print("\n模型评估:")

# Silhouette Score（Silhouette Score）
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# 调整兰德指数（Adjusted Rand Score）
ari = adjusted_rand_score(y, labels)
print(f"调整兰德指数: {ari:.4f}")

# 5. Clustering Results分析
print("\nClustering Results分析:")
print(f"Cluster中心:")
for i, center in enumerate(centers):
    print(f"  Cluster {i}: {center}")

# 统计每个Cluster的样本数
unique, counts = np.unique(labels, return_counts=True)
print("\n每个Cluster的样本数:")
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} 个样本")

# 6. 可视化Clustering Results
print("\n可视化Clustering Results...")

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制Clustering Results
plt.figure(figsize=(12, 6))

# 原始数据
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title('Original Data (True Labels)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='Class')

# Clustering Results
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title(f'K-Means Clustering Results (k={k})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='Cluster')

plt.tight_layout()
plt.savefig('images/kmeans_clustering_results.png')
print("Clustering Results可视化已保存为 'kmeans_clustering_results.png'")

# 7. 确定最佳Cluster数量
print("\n确定最佳Cluster数量...")

inertias = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制Elbow Method和Silhouette Score
plt.figure(figsize=(12, 6))

# Elbow Method
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'o-')
plt.xlabel('Number of Clusters k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(linestyle='--', alpha=0.7)

# Silhouette Score
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'o-')
plt.xlabel('Number of Clusters k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('images/kmeans_elbow_silhouette.png')
print("最佳Cluster数量分析可视化已保存为 'kmeans_elbow_silhouette.png'")

# 8. Clustering Results与真实标签的对应关系
print("\nClustering Results与真实标签的对应关系:")
# 创建Confusion Matrix
confusion_matrix = np.zeros((3, 3), dtype=int)
for true_label, cluster_label in zip(y, labels):
    confusion_matrix[true_label, cluster_label] += 1

print("Confusion Matrix:")
print("真实标签 \\ Cluster标签")
print("\t0\t1\t2")
for i in range(3):
    print(f"{i}\t{confusion_matrix[i, 0]}\t{confusion_matrix[i, 1]}\t{confusion_matrix[i, 2]}")

print("\nscikit-learn K-MeansCluster Demo完成！")
