"""
高级聚类算法（层次聚类、DBSCAN、高斯混合模型）的scikit-learn实现
使用鸢尾花数据集进行聚类分析
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

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

# 3. 层次聚类
print("\n执行层次聚类...")

# 创建层次聚类模型
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')

# 训练模型
agg_labels = agg_clustering.fit_predict(X_scaled)

# 4. DBSCAN聚类
print("\n执行DBSCAN聚类...")

# 创建DBSCAN模型
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 训练模型
dbscan_labels = dbscan.fit_predict(X_scaled)

# 5. 高斯混合模型
print("\n执行高斯混合模型...")

# 创建高斯混合模型
gmm = GaussianMixture(n_components=3, random_state=42)

# 训练模型
gmm_labels = gmm.fit_predict(X_scaled)

# 6. 模型评估
print("\n模型评估:")

# 定义评估函数
def evaluate_model(name, labels):
    """评估聚类模型"""
    # 跳过DBSCAN的评估（可能有噪声点）
    if name == "DBSCAN" and -1 in labels:
        # 只评估非噪声点
        valid_mask = labels != -1
        if np.sum(valid_mask) > 1:
            silhouette = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            ari = adjusted_rand_score(y[valid_mask], labels[valid_mask])
            print(f"{name}:")
            print(f"  轮廓系数: {silhouette:.4f}")
            print(f"  调整兰德指数: {ari:.4f}")
            print(f"  噪声点数量: {np.sum(labels == -1)}")
        else:
            print(f"{name}: 聚类结果无效（噪声点过多）")
    else:
        silhouette = silhouette_score(X_scaled, labels)
        ari = adjusted_rand_score(y, labels)
        print(f"{name}:")
        print(f"  轮廓系数: {silhouette:.4f}")
        print(f"  调整兰德指数: {ari:.4f}")

# 评估各个模型
evaluate_model("层次聚类", agg_labels)
evaluate_model("DBSCAN", dbscan_labels)
evaluate_model("高斯混合模型", gmm_labels)

# 7. 聚类结果分析
print("\n聚类结果分析:")

# 统计每个聚类的样本数
def analyze_clusters(name, labels):
    """分析聚类结果"""
    print(f"\n{name}聚类结果:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        if cluster == -1:
            print(f"  噪声点: {count} 个样本")
        else:
            print(f"  聚类 {cluster}: {count} 个样本")

# 分析各个模型的聚类结果
analyze_clusters("层次聚类", agg_labels)
analyze_clusters("DBSCAN", dbscan_labels)
analyze_clusters("高斯混合模型", gmm_labels)

# 8. 可视化聚类结果
print("\n可视化聚类结果...")

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 绘制聚类结果
plt.figure(figsize=(15, 10))

# 原始数据
plt.subplot(2, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title('原始数据（真实标签）')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='类别')

# 层次聚类结果
plt.subplot(2, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels, cmap='viridis', s=50)
plt.title('层次聚类结果')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='聚类')

# DBSCAN聚类结果
plt.subplot(2, 2, 3)
dbscan_colors = dbscan_labels.copy()
dbscan_colors[dbscan_colors == -1] = 3  # 噪声点设为3
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_colors, cmap='viridis', s=50)
plt.title('DBSCAN聚类结果')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2, 3], label='聚类（3为噪声）')

# 高斯混合模型结果
plt.subplot(2, 2, 4)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title('高斯混合模型结果')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(ticks=[0, 1, 2], label='聚类')

plt.tight_layout()
plt.savefig('images/advanced_clustering_results.png')
print("聚类结果可视化已保存为 'images/advanced_clustering_results.png'")

# 9. 层次聚类的树状图
print("\n绘制层次聚类的树状图...")

plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.savefig('images/hierarchical_clustering_dendrogram.png')
print("层次聚类树状图已保存为 'images/hierarchical_clustering_dendrogram.png'")

# 10. 高斯混合模型的概率分布
print("\n分析高斯混合模型的概率分布...")

# 计算每个样本属于每个聚类的概率
gmm_probs = gmm.predict_proba(X_scaled)

print("高斯混合模型概率分布（前5个样本）:")
for i in range(5):
    print(f"样本 {i}: {gmm_probs[i]}")

# 11. DBSCAN参数调优
print("\nDBSCAN参数调优...")

# 测试不同的eps值
eps_values = [0.3, 0.4, 0.5, 0.6, 0.7]
for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"eps={eps}: 聚类数={n_clusters}, 噪声点数={n_noise}")

print("\nscikit-learn 高级聚类算法 Demo完成！")
