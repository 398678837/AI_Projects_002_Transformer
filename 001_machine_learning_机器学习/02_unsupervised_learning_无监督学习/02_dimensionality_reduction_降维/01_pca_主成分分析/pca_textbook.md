# 主成分分析（PCA）教材

## 第一章：降维基础

### 1.1 什么是降维

降维（Dimensionality Reduction）是将高维数据转换为低维数据的过程，同时尽可能保留数据的重要信息。

#### 1.1.1 为什么需要降维
1. **可视化**：将高维数据降到2D或3D以便可视化
2. **计算效率**：减少特征数量，提高计算速度
3. **减少存储**：减少数据存储空间
4. **去噪**：去除噪声和冗余信息
5. **避免过拟合**：减少特征数量，降低过拟合风险

### 1.2 降维方法分类

#### 1.2.1 线性降维方法
- **主成分分析（PCA）**：最常用的线性降维方法
- **线性判别分析（LDA）**：有监督的线性降维方法
- **因子分析**：寻找潜在因子

#### 1.2.2 非线性降维方法
- **核PCA**：使用核技巧的非线性PCA
- **t-SNE**：t分布邻域嵌入
- **Isomap**：等距映射
- **局部线性嵌入（LLE）**

## 第二章：PCA原理

### 2.1 PCA的基本思想

PCA的目标是找到一组正交基，使得数据在这些基上的投影方差最大。

### 2.2 数学推导

#### 2.2.1 数据中心化
首先对数据进行中心化：

$$ X_c = X - \mu $$

其中 $\mu$ 是数据的均值向量。

#### 2.2.2 协方差矩阵
计算协方差矩阵：

$$ \Sigma = \frac{1}{n-1} X_c^T X_c $$

#### 2.2.3 特征值分解
对协方差矩阵进行特征值分解：

$$ \Sigma v = \lambda v $$

其中：
- $\lambda$ 是特征值
- $v$ 是特征向量

#### 2.2.4 选择主成分
选择最大的K个特征值对应的特征向量作为主成分。

### 2.3 主成分的性质

1. **正交性**：主成分之间是正交的
2. **方差递减**：主成分的方差依次递减
3. **可解释性**：主成分按方差大小排序，第一主成分解释最多的方差

## 第三章：PCA算法流程

### 3.1 算法步骤

1. **数据预处理**：
   - 处理缺失值
   - 特征缩放（标准化）

2. **数据中心化**：
   - 减去均值

3. **计算协方差矩阵**

4. **特征值分解**

5. **选择主成分**：
   - 选择最大的K个特征值对应的特征向量

6. **投影数据**：
   - 将数据投影到主成分上

### 3.2 选择K值

#### 3.2.1 解释方差比
选择使得累计解释方差比达到某个阈值（如95%）的K：

$$ \text{Cumulative Variance Ratio} = \frac{\sum_{k=1}^{K} \lambda_k}{\sum_{k=1}^{D} \lambda_k} $$

#### 3.2.2 碎石图（Scree Plot）
绘制特征值的变化曲线，寻找"肘部"。

## 第四章：PCA实现

### 4.1 纯Python实现

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 存储结果
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        
        # 选择主成分
        if self.n_components is None:
            self.n_components = n_features
        
        self.components_ = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        X = np.array(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

### 4.2 使用scikit-learn

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建PCA模型
pca = PCA(n_components=2)

# 训练和转换
X_pca = pca.fit_transform(X_scaled)

# 查看结果
print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差比: {np.sum(pca.explained_variance_ratio_)}")
print(f"主成分: \n{pca.components_}")
```

## 第五章：PCA可视化

### 5.1 2D可视化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=100, alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.colorbar(scatter, label='Class')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.2 解释方差比可视化

```python
plt.figure(figsize=(12, 5))

# 解释方差比
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio')
plt.grid(True, alpha=0.3)

# 累计解释方差比
plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.legend()

plt.tight_layout()
plt.show()
```

## 第六章：PCA的优缺点

### 6.1 优点

1. **简单高效**：算法原理简单，计算效率高
2. **无监督**：不需要标签信息
3. **去噪**：可以去除噪声和冗余
4. **可解释性**：主成分按方差大小排序，易于解释
5. **应用广泛**：在各种领域都有应用

### 6.2 缺点

1. **线性假设**：假设数据是线性可分的
2. **信息丢失**：可能会丢失一些重要信息
3. **对尺度敏感**：需要进行特征缩放
4. **主成分可能不直观**：主成分可能难以解释
5. **对异常值敏感**：异常值会影响协方差矩阵

## 第七章：PCA的应用

### 7.1 图像压缩

```python
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载人脸数据
faces = fetch_olivetti_faces()
X = faces.data

# 使用PCA降维
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

# 可视化
fig, axes = plt.subplots(2, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(X[i].reshape(64, 64), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(X_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].axis('off')

axes[0, 0].set_title('Original')
axes[1, 0].set_title('Reconstructed')
plt.show()
```

### 7.2 数据可视化

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数字数据
digits = load_digits()
X = digits.data
y = digits.target

# 使用PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=50, alpha=0.7)
plt.colorbar(scatter, label='Digit')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Digits Dataset')
plt.show()
```

## 第八章：总结

### 8.1 核心要点

1. **PCA**是一种线性降维方法，通过最大化投影方差来找到主成分
2. **算法流程**包括数据中心化、协方差矩阵计算、特征值分解、主成分选择和数据投影
3. **选择K值**可以使用解释方差比和碎石图
4. **PCA**在可视化、去噪、压缩等方面有广泛应用
5. **PCA**有线性假设，对异常值敏感

### 8.2 学习路径

1. **基础阶段**：理解PCA的原理和数学推导
2. **实践阶段**：使用Python实现和应用PCA
3. **进阶阶段**：学习核PCA、t-SNE等高级降维方法
4. **应用阶段**：在实际项目中应用降维技术

---

**练习题目**：

1. 证明主成分是正交的。
2. 实现一个简单的PCA算法。
3. 使用PCA对MNIST数据进行降维和可视化。
4. 比较PCA和t-SNE在可视化上的差异。
5. 使用PCA进行图像压缩。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Pattern Recognition and Machine Learning》Bishop
- 《Principal Component Analysis》Jolliffe
