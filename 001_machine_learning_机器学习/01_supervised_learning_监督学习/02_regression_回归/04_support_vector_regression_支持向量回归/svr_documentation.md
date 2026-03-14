# 支持向量回归（Support Vector Regression, SVR）详细文档

## 1. 概念介绍

### 1.1 什么是支持向量回归
支持向量回归（SVR）是支持向量机（SVM）在回归问题上的应用。与SVM分类寻找最大间隔超平面不同，SVR寻找一个超平面，使得尽可能多的数据点位于超平面周围的一个ε-间隔带内。

### 1.2 核心思想
- **ε-间隔带**：允许预测值与真实值之间有一定的误差（ε）
- **支持向量**：位于ε-间隔带边缘或外部的数据点
- **核技巧**：使用核函数将数据映射到高维空间，处理非线性关系
- **正则化**：通过C参数控制模型复杂度

### 1.3 应用场景
- **非线性回归**：当特征与目标变量之间存在非线性关系时
- **小样本学习**：在样本数量较少时表现良好
- **高维数据**：在特征数量较多时表现稳定
- **时间序列预测**：某些时间序列数据的预测
- **金融预测**：股票价格、汇率等预测

## 2. 技术原理

### 2.1 数学模型

#### 2.1.1 线性SVR
线性SVR的优化问题可以表示为：

$$ \min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) $$

约束条件：
$$ y_i - (w^T x_i + b) \leq \epsilon + \xi_i $$
$$ (w^T x_i + b) - y_i \leq \epsilon + \xi_i^* $$
$$ \xi_i, \xi_i^* \geq 0 $$

其中：
- $w$ 是权重向量
- $b$ 是偏置
- $\xi_i, \xi_i^*$ 是松弛变量
- $C$ 是正则化参数
- $\epsilon$ 是间隔带宽度

#### 2.1.2 对偶问题
通过拉格朗日对偶性，可以得到对偶问题：

$$ \min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \epsilon \sum_{i=1}^{n} (\alpha_i + \alpha_i^*) - \sum_{i=1}^{n} y_i (\alpha_i - \alpha_i^*) $$

约束条件：
$$ \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) = 0 $$
$$ 0 \leq \alpha_i, \alpha_i^* \leq C $$

其中 $Q_{ij} = K(x_i, x_j)$ 是核矩阵。

#### 2.1.3 预测函数
SVR的预测函数为：

$$ f(x) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(x_i, x) + b $$

### 2.2 核函数

#### 2.2.1 线性核（Linear）
$$ K(x_i, x_j) = x_i^T x_j $$

适用于线性关系的数据。

#### 2.2.2 多项式核（Polynomial）
$$ K(x_i, x_j) = (\gamma x_i^T x_j + r)^d $$

其中：
- $\gamma$ 是核系数
- $r$ 是常数项
- $d$ 是多项式次数

#### 2.2.3 RBF核（Radial Basis Function）
$$ K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) $$

最常用的核函数，适用于大多数情况。

#### 2.2.4 Sigmoid核
$$ K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r) $$

类似于神经网络的激活函数。

### 2.3 关键参数

#### 2.3.1 C参数
- 控制正则化强度
- C越大，对误差的惩罚越重，模型越复杂
- C越小，对误差的惩罚越轻，模型越简单

#### 2.3.2 ε参数
- 控制ε-间隔带的宽度
- ε越大，允许的误差越大，模型越简单
- ε越小，允许的误差越小，模型越复杂

#### 2.3.3 γ参数（RBF核）
- 控制RBF核的宽度
- γ越大，核函数越窄，模型越复杂
- γ越小，核函数越宽，模型越简单

## 3. 代码实现

### 3.1 scikit-learn实现

文件：`svr_demo.py`

#### 3.1.1 核心步骤
1. **数据准备**：加载或生成数据
2. **数据标准化**：SVR对特征尺度敏感，需要标准化
3. **模型训练**：使用SVR类训练模型
4. **模型评估**：计算MSE、RMSE和R²评分
5. **可视化**：分析支持向量和拟合效果

#### 3.1.2 关键代码

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 创建SVR模型
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

# 训练模型
svr.fit(X_scaled, y_scaled)

# 预测
y_pred_scaled = svr.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 获取支持向量
support_vectors = svr.support_
```

### 3.2 超参数调优

#### 3.2.1 网格搜索
使用网格搜索寻找最佳参数：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1.0, 10.0],
    'epsilon': [0.01, 0.1, 0.2],
    'gamma': ['scale', 'auto', 0.1, 1.0]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_scaled, y_scaled)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")
```

#### 3.2.2 随机搜索
使用随机搜索寻找最佳参数：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {
    'kernel': ['rbf'],
    'C': loguniform(0.1, 100),
    'epsilon': loguniform(0.01, 1),
    'gamma': loguniform(0.01, 10)
}

random_search = RandomizedSearchCV(
    SVR(), param_dist, n_iter=50, cv=5, scoring='r2',
    n_jobs=-1, random_state=42
)
random_search.fit(X_scaled, y_scaled)
```

## 4. 模型评估

### 4.1 回归评估指标

- **均方误差（MSE）**：预测值与真实值之差的平方的平均值
  $$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

- **均方根误差（RMSE）**：MSE的平方根，单位与目标变量相同
  $$ RMSE = \sqrt{MSE} $$

- **平均绝对误差（MAE）**：预测值与真实值之差的绝对值的平均值
  $$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

- **R²评分**：模型解释数据方差的比例
  $$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$
  其中，$\bar{y}$ 是目标变量的均值

### 4.2 支持向量分析

支持向量是位于ε-间隔带边缘或外部的数据点，它们决定了SVR模型的决策函数。

```python
# 获取支持向量的索引
support_indices = svr.support_

# 获取支持向量的数量
n_support = len(support_indices)

# 获取支持向量的系数
dual_coef = svr.dual_coef_
```

## 5. 优缺点分析

### 5.1 优点

- **非线性建模能力强**：通过核函数可以建模复杂的非线性关系
- **小样本学习好**：在样本数量较少时表现良好
- **高维数据稳定**：在特征数量较多时表现稳定
- **泛化能力强**：通过正则化可以有效防止过拟合
- **理论基础扎实**：有坚实的统计学习理论基础

### 5.2 缺点

- **对特征尺度敏感**：需要进行特征标准化
- **训练复杂度高**：时间复杂度为O(n²)，不适合大规模数据
- **参数调优复杂**：需要调优C、ε、γ等多个参数
- **可解释性差**：模型结果难以解释
- **预测速度慢**：预测时需要计算所有支持向量

## 6. 实际应用

### 6.1 房价预测

```python
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 加载数据
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 训练模型
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_scaled, y_scaled)
```

### 6.2 时间序列预测

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 准备时间序列数据
def create_time_series_data(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# 生成数据
t = np.linspace(0, 10, 100)
y = np.sin(t) + np.random.normal(0, 0.1, size=100)

# 创建时间序列特征
X, y_ts = create_time_series_data(y, window_size=5)

# 标准化和训练
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_ts.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_scaled, y_scaled)
```

## 7. 总结

支持向量回归是一种**强大而灵活的回归方法**，它：

1. **基于统计学习理论**：有坚实的理论基础
2. **使用核技巧**：可以建模复杂的非线性关系
3. **通过支持向量确定**：只有支持向量影响模型
4. **需要特征标准化**：对特征尺度敏感
5. **参数调优重要**：需要仔细调优C、ε、γ等参数
6. **小样本表现好**：在样本数量较少时表现良好

SVR虽然在大规模数据上训练较慢，但在中小规模数据上表现优秀，特别是在非线性关系和小样本场景下。通过本文档的学习，你应该已经掌握了SVR的核心概念、实现方法和应用场景，可以开始在实际项目中应用SVR解决各种回归预测问题了。

---

**参考资料**：
- 《统计学习方法》李航
- 《Pattern Recognition and Machine Learning》Christopher M. Bishop
- scikit-learn官方文档
- 《The Nature of Statistical Learning Theory》Vapnik
- 《Support Vector Regression》Drucker et al.
