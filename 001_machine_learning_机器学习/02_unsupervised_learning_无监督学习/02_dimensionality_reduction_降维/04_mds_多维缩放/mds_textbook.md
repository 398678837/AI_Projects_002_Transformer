# MDS降维教材

## 第一章：MDS简介

### 1.1 什么是MDS

MDS通过保持点之间的距离关系来降维。

### 1.2 两种类型

- **度量MDS**：保持距离大小
- **非度量MDS**：保持距离顺序

## 第二章：应力

应力衡量降维后距离与原始距离的差异。

- <0.1: 优秀
- 0.1-0.2: 良好
- >0.2: 一般

## 第三章：实践

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X)
```

## 第四章：总结

MDS适合需要保持距离关系的任务。

---

**参考资料**：
- scikit-learn官方文档
