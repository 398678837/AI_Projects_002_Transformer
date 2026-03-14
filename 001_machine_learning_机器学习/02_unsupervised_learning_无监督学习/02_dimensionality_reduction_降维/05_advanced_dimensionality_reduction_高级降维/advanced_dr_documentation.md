# 高级降维算法（Isomap、LLE）详细文档

## 1. 概念介绍

### 1.1 什么是流形学习
流形学习（Manifold Learning）是一类非线性降维算法，假设高维数据位于一个低维流形上。流形学习的目标是找到这个低维流形，并将数据映射到低维空间。

### 1.2 核心思想
- **流形假设**：高维数据位于一个低维流形上
- **保持局部结构**：保持数据点之间的局部邻域关系
- **非线性降维**：可以处理非线性数据结构

### 1.3 常见算法
- **Isomap**：等度量映射，保持测地线距离
- **LLE**：局部线性嵌入，保持局部线性关系
- **t-SNE**：t分布邻域嵌入，主要用于可视化

## 2. Isomap算法

### 2.1 基本思想
Isomap（Isometric Mapping）的核心思想是保持数据点之间的测地线距离（沿着流形的距离）。

### 2.2 算法流程
1. **构建邻域图**：为每个点找到最近的k个邻居
2. **计算最短路径**：使用Dijkstra算法或Floyd-Warshall算法计算所有点对之间的最短路径
3. **MDS降维**：使用MDS将距离矩阵降维到低维空间

### 2.3 优缺点
**优点**：
- 可以保持全局结构
- 可以展开非线性流形

**缺点**：
- 计算复杂度高O(n³)
- 对噪声敏感
- 对邻域参数k敏感

## 3. LLE算法

### 3.1 基本思想
LLE（Locally Linear Embedding）的核心思想是保持数据点的局部线性关系。

### 3.2 算法流程
1. **构建邻域图**：为每个点找到最近的k个邻居
2. **计算重构权重**：用邻居的线性组合来重构每个点
3. **降维**：在低维空间中保持重构权重不变

### 3.3 优缺点
**优点**：
- 可以保持局部结构
- 计算效率较高
- 可以展开非线性流形

**缺点**：
- 对噪声敏感
- 对邻域参数k敏感
- 可能不能保持全局结构

## 4. 代码实现

### 4.1 Isomap

```python
from sklearn.manifold import Isomap

isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)
```

### 4.2 LLE

```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)
```

## 5. 算法对比

| 算法 | 线性/非线性 | 保持结构 | 计算复杂度 | 对噪声敏感度 |
|------|-------------|---------|-----------|-------------|
| PCA | 线性 | 全局方差 | 低 | 低 |
| Isomap | 非线性 | 全局测地线 | 高 | 中 |
| LLE | 非线性 | 局部 | 中 | 高 |
| t-SNE | 非线性 | 局部 | 高 | 中 |

---

**参考资料**：
- 《Nonlinear Dimensionality Reduction by Locally Linear Embedding》Roweis & Saul
- 《A Global Geometric Framework for Nonlinear Dimensionality Reduction》Tenenbaum et al.
- scikit-learn官方文档
