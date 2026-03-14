# LDA降维教材

## 第一章：LDA简介

### 1.1 什么是LDA

LDA是有监督的降维方法，同时可以用于分类。

### 1.2 LDA vs PCA

- **LDA**：有监督，最大化类别可分性
- **PCA**：无监督，最大化方差

## 第二章：LDA原理

### 2.1 基本思想

找到投影方向，使得：
- 不同类别尽可能远离
- 同一类别尽可能紧凑

### 2.2 重要特点

- 最多降到n_classes-1维
- 需要类别标签
- 可以用于分类

## 第三章：实践

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

## 第四章：总结

LDA在有标签的情况下通常比PCA表现更好。

---

**参考资料**：
- scikit-learn官方文档
