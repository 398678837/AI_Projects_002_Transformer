# MDS（多维缩放）降维详细文档

## 1. 概念介绍

### 1.1 什么是MDS
多维缩放（Multidimensional Scaling，MDS）是一种**降维算法**，其目标是在低维空间中保持高维空间中点之间的距离关系。

### 1.2 核心思想
- **保持距离**：在低维空间中尽可能保持点之间的距离关系
- **应力最小化**：最小化应力（Stress）函数
- **度量vs非度量**：度量MDS保持距离大小，非度量MDS保持距离顺序

### 1.3 应用场景
- **数据可视化**：将高维数据降维到2-3维
- **心理学**：感知数据分析
- **市场研究**：消费者偏好分析
- **生物学**：基因表达数据分析

## 2. 技术原理

### 2.1 应力函数

应力衡量降维后距离与原始距离的差异：

$$ Stress = \sqrt{\frac{\sum_{i<j} (d_{ij} - \hat{d}_{ij})^2}{\sum_{i<j} d_{ij}^2}} $$

其中 $d_{ij}$ 是原始距离，$\hat{d}_{ij}$ 是低维距离。

### 2.2 类型

- **度量MDS**：保持距离大小
- **非度量MDS**：保持距离顺序

## 3. 代码实现

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, metric=True, random_state=42)
X_mds = mds.fit_transform(X)
```

## 4. 优缺点

### 优点
- 直观，保持距离关系
- 不需要假设数据分布
- 度量和非度量两种选择

### 缺点
- 计算复杂度高O(n²)
- 对异常值敏感
- 没有显式映射

---

**参考资料**：
- scikit-learn官方文档
- 《Modern Multidimensional Scaling》Borg & Groenen
