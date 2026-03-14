# 高级降维算法教材

## 第一章：流形学习简介

### 1.1 什么是流形

流形可以理解为"弯曲的平面"，高维数据往往位于一个低维流形上。

### 1.2 为什么需要流形学习

- PCA是线性的，不能处理非线性结构
- 流形学习可以展开非线性流形
- 保持数据的局部或全局结构

## 第二章：Isomap

### 2.1 基本思想

保持测地线距离（沿着流形的距离）。

### 2.2 算法流程

1. 找邻居
2. 计算最短路径
3. MDS降维

### 2.3 特点

- 保持全局结构
- 计算慢（O(n³)）

## 第三章：LLE

### 3.1 基本思想

保持局部线性关系。

### 3.2 算法流程

1. 找邻居
2. 计算重构权重
3. 低维保持权重

### 3.3 特点

- 保持局部结构
- 计算较快
- 对噪声敏感

## 第四章：实践

```python
from sklearn.manifold import Isomap, LocallyLinearEmbedding

isomap = Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)

lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)
```

## 第五章：总结

- Isomap：保持全局，计算慢
- LLE：保持局部，对噪声敏感
- 根据任务选择合适的算法

---

**参考资料**：
- scikit-learn官方文档
