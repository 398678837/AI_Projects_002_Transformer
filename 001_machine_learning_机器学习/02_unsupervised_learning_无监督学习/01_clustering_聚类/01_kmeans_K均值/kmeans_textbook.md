# K-Means聚类教材

## 第一章：聚类分析基础

### 1.1 什么是聚类分析

聚类分析是一种**无监督学习方法**，旨在将相似的数据点分组到同一个簇（cluster）中，同时将不相似的数据点分到不同的簇中。与分类不同，聚类不需要预先标记的数据。

#### 1.1.1 聚类的历史
- **起源**：1930年代，在心理学和人类学领域开始应用
- **发展**：1950年代，随着计算机技术的发展，聚类算法得到广泛研究
- **现代应用**：广泛应用于数据挖掘、模式识别、图像处理、生物信息学等领域

#### 1.1.2 聚类的应用场景
1. **客户细分**：根据消费行为将客户分组
2. **图像分割**：将图像分割为不同区域
3. **基因分析**：识别具有相似表达模式的基因
4. **文档聚类**：将相似的文档分组
5. **异常检测**：识别与其他数据点显著不同的数据

### 1.2 聚类算法的类型

#### 1.2.1 基于划分的聚类
- **K-Means**：最经典的聚类算法
- **K-Medoids**：使用实际数据点作为簇中心
- **K-Modes**：用于类别数据的聚类

#### 1.2.2 层次聚类
- **凝聚聚类**：自底向上，逐步合并簇
- **分裂聚类**：自顶向下，逐步分割簇

#### 1.2.3 基于密度的聚类
- **DBSCAN**：基于密度的空间聚类
- **OPTICS**：排序点以识别聚类结构

#### 1.2.4 基于模型的聚类
- **高斯混合模型（GMM）**：假设数据由多个高斯分布生成
- **期望最大化（EM）算法**：用于估计混合模型参数

## 第二章：K-Means算法原理

### 2.1 K-Means的基本思想

K-Means算法将数据划分为K个簇，使得：
1. 同一簇内的数据点尽可能相似
2. 不同簇之间的数据点尽可能不同

#### 2.1.1 目标函数
K-Means的目标是最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）：

$$ J = \sum_{k=1}^{K} \sum_{i \in C_k} \|x_i - \mu_k\|^2 $$

其中：
- $K$ 是簇的数量
- $C_k$ 是第k个簇
- $x_i$ 是数据点
- $\mu_k$ 是第k个簇的中心（均值）

### 2.2 算法流程

#### 2.2.1 初始化
1. 选择K个初始簇中心
2. 常用的初始化方法：
   - 随机选择K个数据点作为初始中心
   - K-Means++：智能选择初始中心

#### 2.2.2 迭代过程
重复以下步骤直到收敛：
1. **分配步骤**：将每个数据点分配到最近的簇中心
2. **更新步骤**：重新计算每个簇的中心（均值）

#### 2.2.3 收敛条件
- 簇分配不再变化
- 或簇中心的变化小于某个阈值
- 或达到最大迭代次数

### 2.3 距离度量

#### 2.3.1 欧几里得距离
$$ d(x_i, x_j) = \sqrt{\sum_{d=1}^{D} (x_{id} - x_{jd})^2} $$

#### 2.3.2 曼哈顿距离
$$ d(x_i, x_j) = \sum_{d=1}^{D} |x_{id} - x_{jd}| $$

#### 2.3.3 余弦相似度
$$ \text{sim}(x_i, x_j) = \frac{x_i \cdot x_j}{\|x_i\| \|x_j\|} $$

## 第三章：K-Means算法实现

### 3.1 纯Python实现

```python
import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # 初始化簇中心
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 随机选择K个数据点作为初始中心
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices]
        
        for iteration in range(self.max_iter):
            # 分配步骤：将每个数据点分配到最近的簇中心
            distances = self._calculate_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            # 更新步骤：重新计算簇中心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    # 如果簇为空，随机选择一个新的中心
                    new_centroids[k] = X[np.random.choice(n_samples)]
            
            # 检查收敛
            centroid_change = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.labels_ = labels
            
            if centroid_change < self.tol:
                break
        
        # 计算inertia（WCSS）
        self.inertia_ = self._calculate_inertia(X)
        return self
    
    def predict(self, X):
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X, centroids):
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        
        return distances
    
    def _calculate_inertia(self, X):
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                inertia += np.sum(np.linalg.norm(cluster_points - self.centroids[k], axis=1)**2)
        return inertia
```

### 3.2 使用scikit-learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建K-Means模型
kmeans = KMeans(n_clusters=4, random_state=0)

# 训练模型
kmeans.fit(X)

# 预测
y_pred = kmeans.predict(X)

# 获取簇中心
centroids = kmeans.cluster_centers_

# 获取inertia
inertia = kmeans.inertia_
```

## 第四章：选择K值

### 4.1 肘部法则（Elbow Method）

#### 4.1.1 原理
绘制不同K值下的inertia曲线，寻找曲线的"肘部"点。

#### 4.1.2 实现
```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.show()
```

### 4.2 轮廓系数（Silhouette Score）

#### 4.2.1 原理
轮廓系数衡量数据点与其所在簇内其他点的相似度，以及与其他簇内点的不相似度：

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

其中：
- $a(i)$ 是数据点i与同簇内其他点的平均距离
- $b(i)$ 是数据点i与最近的其他簇内所有点的平均距离

#### 4.2.2 解释
- $s(i) \approx 1$：数据点与所在簇匹配得很好
- $s(i) \approx 0$：数据点在两个簇的边界上
- $s(i) \approx -1$：数据点被分到了错误的簇

#### 4.2.3 实现
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(K_range, silhouette_scores, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.grid(True)
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print(f"最佳K值: {best_k}")
```

### 4.3 间隙统计量（Gap Statistic）

间隙统计量比较观察到的inertia与参考分布（通常是均匀分布）的inertia。

```python
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def calculate_gap_statistic(X, K_range, n_references=5):
    gaps = np.zeros(len(K_range))
    gaps_std = np.zeros(len(K_range))
    
    for i, k in enumerate(K_range):
        # 计算观察到的inertia
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)
        inertia_obs = kmeans.inertia_
        
        # 计算参考分布的inertia
        inertia_refs = []
        for _ in range(n_references):
            X_ref = np.random.uniform(X.min(), X.max(), size=X.shape)
            kmeans_ref = KMeans(n_clusters=k, random_state=0)
            kmeans_ref.fit(X_ref)
            inertia_refs.append(kmeans_ref.inertia_)
        
        # 计算间隙统计量
        gaps[i] = np.mean(np.log(inertia_refs)) - np.log(inertia_obs)
        gaps_std[i] = np.sqrt(1 + 1/n_references) * np.std(np.log(inertia_refs))
    
    return gaps, gaps_std
```

## 第五章：K-Means++初始化

### 5.1 为什么需要K-Means++

传统的随机初始化可能导致：
- 收敛到局部最优解
- 结果不稳定
- 需要多次运行选择最佳结果

### 5.2 K-Means++算法流程

1. 从数据集中随机选择一个点作为第一个簇中心
2. 对于每个数据点，计算其到最近的已选簇中心的距离
3. 选择一个新的数据点作为簇中心，概率与距离平方成正比
4. 重复步骤2-3，直到选择K个簇中心
5. 运行标准的K-Means算法

### 5.3 K-Means++的优点

- 提高收敛速度
- 获得更好的聚类结果
- 减少对初始值的敏感性

## 第六章：数据预处理

### 6.1 特征缩放

K-Means对特征尺度敏感，因此需要进行特征缩放：

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 6.2 处理类别特征

对于类别特征，需要进行编码：

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
X_encoded = encoder.fit_transform(X_categorical)
```

### 6.3 处理缺失值

```python
from sklearn.impute import SimpleImputer, KNNImputer

# 使用均值填充
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 使用KNN填充
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

## 第七章：可视化聚类结果

### 7.1 2D可视化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, linewidths=3, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Results')
plt.legend()
plt.colorbar(label='Cluster')
plt.show()
```

### 7.2 高维数据可视化

使用PCA或t-SNE将高维数据降维到2D或3D：

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)
```

## 第八章：K-Means的优缺点

### 8.1 优点

1. **简单易懂**：算法原理简单，易于实现
2. **计算高效**：时间复杂度为O(n*K*t)，其中t是迭代次数
3. **可扩展**：适用于大规模数据集
4. **应用广泛**：在实际问题中表现良好
5. **易于解释**：结果直观，易于理解

### 8.2 缺点

1. **需要预先指定K**：K值的选择影响聚类结果
2. **对初始值敏感**：不同的初始值可能导致不同的结果
3. **对异常值敏感**：异常值会影响簇中心
4. **假设簇是球形的**：不适用于非球形簇
5. **对特征尺度敏感**：需要进行特征缩放

## 第九章：实际应用

### 9.1 客户细分

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('customer_data.csv')
X = data[['recency', 'frequency', 'monetary']]

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用肘部法则选择K
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 训练K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
data['cluster'] = kmeans.fit_predict(X_scaled)

# 分析每个簇
cluster_analysis = data.groupby('cluster').agg({
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
})
print(cluster_analysis)
```

### 9.2 图像压缩

```python
from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

# 加载图像
image = Image.open('image.jpg')
X = np.array(image).reshape(-1, 3)

# 使用K-Means压缩
kmeans = KMeans(n_clusters=16, random_state=0)
kmeans.fit(X)

# 替换像素值
X_compressed = kmeans.cluster_centers_[kmeans.labels_]
X_compressed = X_compressed.reshape(image.shape)

# 保存压缩后的图像
Image.fromarray(X_compressed.astype(np.uint8)).save('image_compressed.jpg')
```

## 第十章：总结

### 10.1 核心要点

1. **K-Means**是一种经典的基于划分的聚类算法
2. **目标是最小化簇内平方和**
3. **算法流程**包括初始化、分配、更新三个步骤
4. **选择K值**可以使用肘部法则、轮廓系数、间隙统计量等方法
5. **K-Means++**可以提高初始化质量
6. **数据预处理**对K-Means至关重要

### 10.2 学习路径

1. **基础阶段**：理解K-Means的原理和实现
2. **实践阶段**：使用Python实现K-Means
3. **进阶阶段**：学习K-Means++、特征选择等高级技术
4. **应用阶段**：在实际项目中应用K-Means

### 10.3 进一步学习

- **层次聚类**：学习层次聚类方法
- **DBSCAN**：学习基于密度的聚类
- **高斯混合模型**：学习基于模型的聚类
- **谱聚类**：学习谱聚类方法
- **聚类评估**：深入学习聚类评估指标

---

**练习题目**：

1. 证明K-Means算法在每次迭代中都会单调减小目标函数。
2. 实现K-Means++初始化算法。
3. 设计一个实验比较不同初始化方法的效果。
4. 实现一个自动选择K值的函数。
5. 使用K-Means对MNIST数据进行聚类分析。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Data Mining: Concepts and Techniques》Han, Kamber, Pei
- 《K-Means++: The Advantages of Careful Seeding》Arthur, Vassilvitskii
