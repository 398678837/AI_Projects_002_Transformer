# 多项式回归（Polynomial Regression）详细文档

## 1. 概念介绍

### 1.1 什么是多项式回归
多项式回归是线性回归的扩展，通过添加特征的多项式项来建模特征与目标变量之间的非线性关系。虽然模型在参数上是线性的，但可以拟合非线性数据。

### 1.2 核心思想
- **非线性建模**：通过引入多项式项，可以捕捉特征与目标变量之间的非线性关系
- **线性参数**：虽然特征是非线性的，但模型在参数上仍然是线性的
- **过拟合风险**：随着多项式次数的增加，模型可能会过拟合训练数据

### 1.3 应用场景
- **非线性关系建模**：当特征与目标变量之间存在明显的非线性关系时
- **曲线拟合**：需要拟合曲线而不是直线时
- **时间序列预测**：某些时间序列数据可能呈现非线性趋势
- **科学实验数据**：物理、化学等实验数据的拟合

## 2. 技术原理

### 2.1 数学模型

#### 2.1.1 简单线性回归
$$ y = \beta_0 + \beta_1 x + \epsilon $$

#### 2.1.2 二次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \epsilon $$

#### 2.1.3 三次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \epsilon $$

#### 2.1.4 d次多项式回归
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d + \epsilon $$

### 2.2 多元多项式回归

对于多个特征，多项式回归可以包含特征之间的交互项：

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 x_1 x_2 + \epsilon $$

### 2.3 特征变换

多项式回归可以看作是特征变换后的线性回归：

1. **原始特征**：$x$
2. **变换后的特征**：$[x, x^2, x^3, \cdots, x^d]$
3. **线性回归**：在变换后的特征上应用线性回归

### 2.4 过拟合与欠拟合

#### 2.4.1 欠拟合（Underfitting）
- 多项式次数太低
- 模型过于简单，无法捕捉数据中的模式
- 训练集和测试集的误差都很高

#### 2.4.2 过拟合（Overfitting）
- 多项式次数太高
- 模型过于复杂，学习了训练数据中的噪声
- 训练集误差低，但测试集误差高

#### 2.4.3 适度拟合（Good Fit）
- 多项式次数适中
- 模型能够捕捉数据中的真实模式
- 训练集和测试集的误差都较低

## 3. 代码实现

### 3.1 scikit-learn实现

文件：`polynomial_regression_demo.py`

#### 3.1.1 核心步骤
1. **数据准备**：加载或生成数据
2. **特征变换**：使用PolynomialFeatures创建多项式特征
3. **模型训练**：在变换后的特征上训练线性回归模型
4. **模型评估**：计算MSE、RMSE和R²评分
5. **可视化**：分析多项式拟合效果

#### 3.1.2 关键代码

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# 创建Pipeline
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear', LinearRegression())
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)
```

### 3.2 选择多项式次数

#### 3.2.1 验证集方法
使用验证集选择最佳的多项式次数：

```python
from sklearn.model_selection import train_test_split

# 划分训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

degrees = range(1, 10)
val_r2_scores = []

for degree in degrees:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    val_r2_scores.append(r2_score(y_val, pipeline.predict(X_val)))

best_degree = degrees[np.argmax(val_r2_scores)]
print(f"最佳多项式次数: {best_degree}")
```

#### 3.2.2 交叉验证方法
使用K折交叉验证选择最佳的多项式次数：

```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 10)
cv_r2_scores = []

for degree in degrees:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    cv_r2_scores.append(scores.mean())

best_degree = degrees[np.argmax(cv_r2_scores)]
print(f"最佳多项式次数: {best_degree}")
```

## 4. 正则化多项式回归

### 4.1 为什么需要正则化
多项式回归容易过拟合，特别是当多项式次数较高时。正则化可以帮助控制模型复杂度，防止过拟合。

### 4.2 Ridge多项式回归

```python
from sklearn.linear_model import Ridge

pipeline_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0))
])

pipeline_ridge.fit(X_train, y_train)
y_pred = pipeline_ridge.predict(X_test)
```

### 4.3 Lasso多项式回归

```python
from sklearn.linear_model import Lasso

pipeline_lasso = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lasso', Lasso(alpha=0.1))
])

pipeline_lasso.fit(X_train, y_train)
y_pred = pipeline_lasso.predict(X_test)
```

### 4.4 ElasticNet多项式回归

```python
from sklearn.linear_model import ElasticNet

pipeline_enet = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('elasticnet', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

pipeline_enet.fit(X_train, y_train)
y_pred = pipeline_enet.predict(X_test)
```

## 5. 模型评估

### 5.1 回归评估指标

- **均方误差（MSE）**：预测值与真实值之差的平方的平均值
  $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

- **均方根误差（RMSE）**：MSE的平方根，单位与目标变量相同
  $$ RMSE = \sqrt{MSE} $$

- **平均绝对误差（MAE）**：预测值与真实值之差的绝对值的平均值
  $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

- **R²评分**：模型解释数据方差的比例
  $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$
  其中，$\bar{y}$ 是目标变量的均值

### 5.2 学习曲线

学习曲线可以帮助我们诊断过拟合和欠拟合：

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='训练集')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.plot(train_sizes, val_mean, 's-', label='验证集')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('训练样本数')
plt.ylabel('R² Score')
plt.title('学习曲线')
plt.legend()
plt.show()
```

## 6. 优缺点分析

### 6.1 优点

- **简单直观**：算法原理简单，容易理解和实现
- **灵活建模**：可以建模各种非线性关系
- **可解释性强**：可以解释特征对目标变量的影响
- **计算高效**：训练速度快，适合处理中等规模数据
- **扩展性强**：可以与正则化方法结合使用

### 6.2 缺点

- **过拟合风险**：多项式次数过高时容易过拟合
- **特征爆炸**：多项式次数和特征数量增加时，特征数量会急剧增加
- **外推能力差**：在训练数据范围之外的预测可能不准确
- **需要调参**：需要选择合适的多项式次数
- **对异常值敏感**：异常值会对模型产生较大影响

## 7. 实际应用

### 7.1 房价预测

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

### 7.2 时间序列预测

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

## 8. 总结

多项式回归是一种**强大而灵活的回归方法**，它：

1. **可以建模非线性关系**：通过多项式项捕捉特征与目标变量之间的非线性关系
2. **本质上是线性回归**：在变换后的特征上应用线性回归
3. **容易过拟合**：需要选择合适的多项式次数
4. **可以与正则化结合**：使用Ridge、Lasso或ElasticNet防止过拟合
5. **应用广泛**：可以应用于各种非线性回归预测任务

多项式回归虽然简单，但它是许多复杂非线性模型的基础。通过本文档的学习，你应该已经掌握了多项式回归的核心概念、实现方法和应用场景，可以开始在实际项目中应用多项式回归解决各种非线性回归预测问题了。

---

**参考资料**：
- 《统计学习方法》李航
- 《Pattern Recognition and Machine Learning》Christopher M. Bishop
- scikit-learn官方文档
- 《Introduction to Statistical Learning》Gareth James et al.
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
