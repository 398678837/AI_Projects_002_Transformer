"""
K-Means聚类算法的scikit-learn实现
使用鸢尾花数据集进行聚类分析
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
print(f"类别名称: {iris.target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-Means聚类
print("\n执行K-Means聚类...")
# 设置聚类数量为3（与真实类别数一致）
k = 3

# 创建K-Means模型
kmeans = KMeans(n_clusters=k, random_state=42)

# 训练模型
kmeans.fit(X_scaled)

# 获取聚类结果
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 4. 模型评估
print("\n模型评估:")

# 轮廓系数（Silhouette Score）
silhouette = silhouette_score(X_scaled, labels)
print(f"轮廓系数: {silhouette:.4f}")

# 调整兰德指数（Adjusted Rand Score）
ari = adjusted_rand_score(y, labels)
print(f"调整兰德指数: {ari:.4f}")

# 5. 聚类结果分析
print("\n聚类结果分析:")
print(f"聚类中心:")
for i, center in enumerate(centers):
    print(f"  聚类 {i}: {center}")

# 统计每个聚类的样本数
unique, counts = np.unique(labels, return_counts=True)
print("\n每个聚类的样本数:")
for cluster, count in zip(unique, counts):
    print(f"  聚类 {cluster}: {count} 个样本")

# 6. 可视化聚类结果
print("\n可视化聚类结果...")

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制聚类结果
plt.figure(figsize=(12, 6))

# 原始数据
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title('原始数据（真实标签）')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='类别')

# 聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title(f'K-Means聚类结果 (k={k})')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='聚类')

plt.tight_layout()
plt.savefig('kmeans_clustering_results.png')
print("聚类结果可视化已保存为 'kmeans_clustering_results.png'")

# 7. 确定最佳聚类数量
print("\n确定最佳聚类数量...")

inertias = []
silhouette_scores = []
K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制肘部法则和轮廓系数
plt.figure(figsize=(12, 6))

# 肘部法则
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'o-')
plt.xlabel('聚类数量 k')
plt.ylabel('惯性（Inertia）')
plt.title('肘部法则')
plt.grid(linestyle='--', alpha=0.7)

# 轮廓系数
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'o-')
plt.xlabel('聚类数量 k')
plt.ylabel('轮廓系数')
plt.title('轮廓系数')
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('kmeans_elbow_silhouette.png')
print("最佳聚类数量分析可视化已保存为 'kmeans_elbow_silhouette.png'")

# 8. 聚类结果与真实标签的对应关系
print("\n聚类结果与真实标签的对应关系:")
# 创建混淆矩阵
confusion_matrix = np.zeros((3, 3), dtype=int)
for true_label, cluster_label in zip(y, labels):
    confusion_matrix[true_label, cluster_label] += 1

print("混淆矩阵:")
print("真实标签 \\ 聚类标签")
print("\t0\t1\t2")
for i in range(3):
    print(f"{i}\t{confusion_matrix[i, 0]}\t{confusion_matrix[i, 1]}\t{confusion_matrix[i, 2]}")

print("\nscikit-learn K-Means聚类 Demo完成！")
