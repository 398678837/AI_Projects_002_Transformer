# 高级聚类教材

## 第一章：层次聚类

### 1.1 什么是层次聚类

层次聚类（Hierarchical Clustering）是一种通过构建簇的层次结构来进行聚类的方法。它不需要预先指定簇的数量。

#### 1.1.1 层次聚类的类型
1. **凝聚聚类（Agglomerative）**：自底向上，初始时每个点都是一个簇，逐步合并相似的簇
2. **分裂聚类（Divisive）**：自顶向下，初始时所有点属于一个簇，逐步分割

### 1.2 凝聚聚类

#### 1.2.1 算法流程
1. **初始化**：每个数据点作为一个簇
2. **合并**：找到最相似的两个簇，将它们合并
3. **更新**：更新簇之间的相似度
4. **重复**：重复步骤2-3，直到所有点合并为一个簇

#### 1.2.2 链接方法
1. **单链接（Single Linkage）**：两个簇中最近点之间的距离
2. **完全链接（Complete Linkage）**：两个簇中最远点之间的距离
3. **平均链接（Average Linkage）**：两个簇中所有点对之间的平均距离
4. **质心链接（Centroid Linkage）**：两个簇质心之间的距离
5. **Ward链接**：最小化簇内平方和的增加

### 1.3 树状图（Dendrogram）

树状图是层次聚类结果的可视化：

```python
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# 计算链接矩阵
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15.)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
```

## 第二章：DBSCAN

### 2.1 什么是DBSCAN

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，它可以发现任意形状的簇，并且可以识别噪声点。

### 2.2 核心概念

#### 2.2.1 核心点
如果一个点的ε邻域内至少有min_samples个点（包括自身），则该点是核心点。

#### 2.2.2 边界点
不是核心点，但位于某个核心点的ε邻域内。

#### 2.2.3 噪声点
既不是核心点，也不是边界点。

#### 2.2.4 直接密度可达
如果点p在核心点q的ε邻域内，则p从q直接密度可达。

#### 2.2.5 密度可达
如果存在一条路径p1, p2, ..., pn，其中p1=p, pn=q，且每个pi+1从pi直接密度可达，则q从p密度可达。

#### 2.2.6 密度相连
如果存在一个点o，使得p和q都从o密度可达，则p和q密度相连。

### 2.3 算法流程

1. **初始化**：所有点未标记
2. **对于每个未标记的点p**：
   - 找到p的ε邻域内的所有点
   - 如果邻域内点数小于min_samples，标记为噪声
   - 否则，创建一个新簇，将p标记为核心点，将邻域内所有点加入队列
3. **处理队列中的点**：
   - 对于队列中的每个点q：
     - 如果q未标记，标记为当前簇
     - 找到q的ε邻域内的所有点
     - 如果q是核心点，将邻域内未标记的点加入队列
4. **重复**：直到所有点都被标记

### 2.4 Python实现

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 生成数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

# 创建DBSCAN模型
dbscan = DBSCAN(eps=0.3, min_samples=5)

# 训练
y_pred = dbscan.fit_predict(X)

# 获取核心点样本
core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# 可视化
plt.figure(figsize=(10, 8))

# 绘制簇
unique_labels = set(y_pred)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点
        col = [0, 0, 0, 1]
    
    class_member_mask = (y_pred == k)
    
    # 核心点
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
    
    # 非核心点
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('DBSCAN Clustering')
plt.show()
```

## 第三章：高斯混合模型（GMM）

### 3.1 什么是高斯混合模型

高斯混合模型（Gaussian Mixture Model, GMM）假设数据是由多个高斯分布生成的，每个高斯分布对应一个簇。

### 3.2 模型定义

GMM的概率密度函数为：

$$ p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) $$

其中：
- $K$ 是高斯分布的数量
- $\pi_k$ 是第k个高斯分布的权重（$\sum_{k=1}^{K} \pi_k = 1$）
- $\mathcal{N}(x | \mu_k, \Sigma_k)$ 是第k个高斯分布

### 3.3 期望最大化（EM）算法

#### 3.3.1 E步（期望步）
计算每个数据点属于每个簇的后验概率：

$$ \gamma(z_{ik}) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)} $$

#### 3.3.2 M步（最大化步）
更新参数：

$$ N_k = \sum_{i=1}^{n} \gamma(z_{ik}) $$

$$ \pi_k = \frac{N_k}{n} $$

$$ \mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma(z_{ik}) x_i $$

$$ \Sigma_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma(z_{ik}) (x_i - \mu_k)(x_i - \mu_k)^T $$

### 3.4 Python实现

```python
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建GMM模型
gmm = GaussianMixture(n_components=4, random_state=0)

# 训练
gmm.fit(X)

# 预测
y_pred = gmm.predict(X)

# 获取概率
probs = gmm.predict_proba(X)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Mixture Model Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

## 第四章：聚类评估

### 4.1 内部评估指标

#### 4.1.1 轮廓系数（Silhouette Score）

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(f"Silhouette Score: {score:.4f}")
```

#### 4.1.2 Calinski-Harabasz指数

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Score: {score:.4f}")
```

#### 4.1.3 Davies-Bouldin指数

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Score: {score:.4f}")
```

### 4.2 外部评估指标

#### 4.2.1 调整兰德指数（Adjusted Rand Index）

```python
from sklearn.metrics import adjusted_rand_score

score = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index: {score:.4f}")
```

#### 4.2.2 归一化互信息（Normalized Mutual Information）

```python
from sklearn.metrics import normalized_mutual_info_score

score = normalized_mutual_info_score(y_true, y_pred)
print(f"Normalized Mutual Information: {score:.4f}")
```

## 第五章：聚类算法比较

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| K-Means | 简单、快速、可扩展 | 需要指定K，假设球形簇，对初始值敏感 | 球形簇、大数据集 |
| 层次聚类 | 不需要指定K，可视化树状图 | 计算复杂度高，对噪声敏感 | 小数据集、需要层次结构 |
| DBSCAN | 发现任意形状，识别噪声 | 需要调优ε和min_samples，对密度变化敏感 | 任意形状、有噪声 |
| GMM | 概率模型，软聚类 | 需要指定K，收敛到局部最优 | 软聚类、高斯分布假设 |

## 第六章：总结

### 6.1 核心要点

1. **层次聚类**通过构建簇的层次结构进行聚类
2. **DBSCAN**基于密度，可以发现任意形状的簇
3. **GMM**假设数据由多个高斯分布生成，提供软聚类
4. **聚类评估**包括内部指标和外部指标
5. **没有万能的算法**，需要根据数据特点选择合适的算法

### 6.2 学习路径

1. **基础阶段**：理解各种聚类算法的原理
2. **实践阶段**：使用Python实现和应用各种聚类算法
3. **进阶阶段**：学习聚类评估和算法调优
4. **应用阶段**：在实际项目中应用聚类算法

---

**练习题目**：

1. 实现一个简单的层次聚类算法。
2. 比较不同链接方法在层次聚类中的效果。
3. 实现DBSCAN算法。
4. 使用GMM对MNIST数据进行聚类。
5. 设计一个实验比较各种聚类算法的性能。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Pattern Recognition and Machine Learning》Bishop
- 《Density-Based Spatial Clustering of Applications with Noise》Ester et al.
