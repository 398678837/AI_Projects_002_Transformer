# 多项式回归教材

## 第一章：回归分析基础

### 1.1 线性回归回顾

线性回归是一种基础的监督学习方法，用于预测连续型目标变量。它假设目标变量与特征之间存在线性关系：

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon $$

### 1.2 线性回归的局限性

线性回归的主要局限性是它只能建模线性关系。在现实世界中，许多关系是非线性的。

#### 1.2.1 非线性关系的例子
1. **物理定律**：如自由落体运动 $s = \frac{1}{2}gt^2$
2. **经济增长**：通常呈现指数增长
3. **生物生长**：如人口增长、细菌繁殖
4. **化学反应**：反应速率与温度的关系

## 第二章：多项式回归原理

### 2.1 什么是多项式回归

多项式回归是线性回归的扩展，通过添加特征的多项式项来建模非线性关系。

#### 2.1.1 二次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon $$

#### 2.1.2 三次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \epsilon $$

#### 2.1.3 d次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \epsilon $$

### 2.2 多项式回归的本质

虽然多项式回归可以建模非线性关系，但它在参数上仍然是线性的。这意味着我们可以使用线性回归的方法来求解多项式回归。

#### 2.2.1 特征变换
将原始特征 $x$ 变换为多项式特征：
$$ z_1 = x, \quad z_2 = x^2, \quad z_3 = x^3, \quad \cdots, \quad z_d = x^d $$

然后在变换后的特征上应用线性回归：
$$ y = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + \cdots + \beta_d z_d + \epsilon $$

### 2.3 多元多项式回归

对于多个特征，多项式回归可以包含特征之间的交互项。

#### 2.3.1 两个特征的二次多项式回归
$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \epsilon $$

#### 2.3.2 交互项的意义
交互项 $x_1 x_2$ 表示特征 $x_1$ 和 $x_2$ 之间的交互效应。

## 第三章：过拟合与欠拟合

### 3.1 什么是过拟合和欠拟合

#### 3.1.1 欠拟合（Underfitting）
- 模型过于简单，无法捕捉数据中的模式
- 训练集和测试集的误差都很高
- 高偏差，低方差

#### 3.1.2 过拟合（Overfitting）
- 模型过于复杂，学习了训练数据中的噪声
- 训练集误差低，但测试集误差高
- 低偏差，高方差

#### 3.1.3 适度拟合（Good Fit）
- 模型复杂度适中，能够捕捉数据中的真实模式
- 训练集和测试集的误差都较低
- 偏差和方差的平衡

### 3.2 多项式次数的选择

选择合适的多项式次数是多项式回归的关键。

#### 3.2.1 验证集方法
将数据划分为训练集、验证集和测试集：
1. 在训练集上训练不同次数的多项式模型
2. 在验证集上评估模型性能
3. 选择在验证集上性能最好的模型
4. 在测试集上评估最终模型

#### 3.2.2 交叉验证方法
使用K折交叉验证：
1. 将数据划分为K个大小相等的子集
2. 对于每个子集：
   - 使用该子集作为验证集
   - 使用其他K-1个子集作为训练集
   - 训练模型并在验证集上评估
3. 计算K次评估的平均性能
4. 选择平均性能最好的模型

### 3.3 学习曲线

学习曲线可以帮助我们诊断过拟合和欠拟合。

#### 3.3.1 绘制学习曲线
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)
)
```

#### 3.3.2 解读学习曲线
- **欠拟合**：训练集和验证集的曲线都很低，且接近
- **过拟合**：训练集的曲线很高，但验证集的曲线很低，两者差距大
- **适度拟合**：训练集和验证集的曲线都较高，且接近

## 第四章：正则化多项式回归

### 4.1 为什么需要正则化

多项式回归容易过拟合，特别是当多项式次数较高时。正则化可以帮助控制模型复杂度，防止过拟合。

### 4.2 Ridge回归

Ridge回归通过添加L2正则化项来控制模型复杂度：

$$ \min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2 $$

#### 4.2.1 alpha的选择
- $\alpha = 0$：等价于普通线性回归
- $\alpha$ 增大：模型复杂度降低，可能欠拟合
- $\alpha$ 减小：模型复杂度增加，可能过拟合

### 4.3 Lasso回归

Lasso回归通过添加L1正则化项来控制模型复杂度：

$$ \min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j| $$

#### 4.3.1 Lasso的特点
- 可以进行特征选择：将某些系数压缩到0
- 适用于特征稀疏的情况
- 对高相关性的特征敏感

### 4.4 ElasticNet

ElasticNet结合了L1和L2正则化：

$$ \min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \alpha \left( \rho \sum_{j=1}^{p} |\beta_j| + (1-\rho) \sum_{j=1}^{p} \beta_j^2 \right) $$

#### 4.4.1 l1_ratio的选择
- `l1_ratio = 1`：等价于Lasso
- `l1_ratio = 0`：等价于Ridge
- `0 < l1_ratio < 1`：结合L1和L2正则化

## 第五章：多项式回归实现

### 5.1 使用scikit-learn

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 创建Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

# 训练
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

### 5.2 纯Python实现

```python
import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coefficients = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        # 创建多项式特征
        X_poly = self._create_polynomial_features(X)
        
        # 添加截距项
        X_poly = np.c_[np.ones(len(X_poly)), X_poly]
        
        # 使用正规方程求解
        self.coefficients = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        
        return self
    
    def predict(self, X):
        X = np.array(X)
        
        # 创建多项式特征
        X_poly = self._create_polynomial_features(X)
        
        # 添加截距项
        X_poly = np.c_[np.ones(len(X_poly)), X_poly]
        
        return X_poly @ self.coefficients
    
    def _create_polynomial_features(self, X):
        n_samples, n_features = X.shape
        X_poly = []
        
        for i in range(n_samples):
            features = []
            for d in range(1, self.degree + 1):
                for j in range(n_features):
                    features.append(X[i, j] ** d)
            X_poly.append(features)
        
        return np.array(X_poly)
```

## 第六章：实际应用

### 6.1 房价预测

```python
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 使用Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

# 训练
pipeline.fit(X, y)
```

### 6.2 时间序列预测

```python
import pandas as pd
import numpy as np

# 创建时间序列特征
def create_time_features(t, degree=3):
    features = []
    for d in range(1, degree + 1):
        features.append(t ** d)
    return np.array(features).T

# 生成时间序列
t = np.linspace(0, 10, 100)
y = np.sin(t) + np.random.normal(0, 0.1, size=100)

# 创建多项式特征
X = create_time_features(t, degree=3)

# 训练模型
model = LinearRegression()
model.fit(X, y)
```

## 第七章：总结

### 7.1 核心要点

1. **多项式回归**是线性回归的扩展，可以建模非线性关系
2. **在参数上仍然是线性的**，可以使用线性回归的方法求解
3. **容易过拟合**，需要选择合适的多项式次数
4. **可以与正则化结合**，使用Ridge、Lasso或ElasticNet防止过拟合
5. **需要权衡偏差和方差**，找到适度拟合的模型

### 7.2 学习路径

1. **基础阶段**：理解线性回归和多项式回归的原理
2. **实践阶段**：使用Python实现多项式回归
3. **进阶阶段**：学习正则化方法和模型选择
4. **应用阶段**：在实际项目中应用多项式回归

---

**练习题目**：

1. 证明多项式回归在参数上是线性的。
2. 实现一个简单的多项式回归算法。
3. 使用验证集方法选择最佳的多项式次数。
4. 比较Ridge、Lasso和ElasticNet在多项式回归中的效果。
5. 使用多项式回归对一个时间序列进行预测。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Pattern Recognition and Machine Learning》Bishop
- 《Introduction to Statistical Learning》Gareth James et al.
