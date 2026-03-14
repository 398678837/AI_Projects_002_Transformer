# 支持向量回归教材

## 第一章：支持向量机回顾

### 1.1 什么是支持向量机

支持向量机（Support Vector Machine, SVM）是一种监督学习方法，最初用于分类问题，后来扩展到回归问题。

#### 1.1.1 SVM的核心思想
- **最大间隔**：寻找能够最大化两类之间间隔的超平面
- **支持向量**：位于间隔边缘的数据点，决定了超平面的位置
- **核技巧**：使用核函数将数据映射到高维空间，处理非线性问题

### 1.2 从分类到回归

SVM分类寻找最大间隔超平面，SVR寻找一个超平面，使得尽可能多的数据点位于超平面周围的一个ε-间隔带内。

## 第二章：SVR原理

### 2.1 ε-间隔带

SVR引入了ε-间隔带的概念：
- 如果预测值与真实值的差小于ε，则不计入误差
- 只有当误差超过ε时，才会产生惩罚

#### 2.1.1 数学表示
对于每个数据点 $(x_i, y_i)$，我们允许：
$$ |y_i - f(x_i)| \leq \epsilon $$

其中 $f(x)$ 是预测函数。

### 2.2 优化问题

#### 2.2.1 原始问题
线性SVR的优化问题可以表示为：

$$ \min_{w, b, \xi, \xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) $$

约束条件：
$$ y_i - (w^T x_i + b) \leq \epsilon + \xi_i $$
$$ (w^T x_i + b) - y_i \leq \epsilon + \xi_i^* $$
$$ \xi_i, \xi_i^* \geq 0 $$

#### 2.2.2 参数解释
- $w$：权重向量
- $b$：偏置
- $\xi_i, \xi_i^*$：松弛变量，允许数据点位于ε-间隔带之外
- $C$：正则化参数，控制对误差的惩罚程度
- $\epsilon$：间隔带宽度，控制允许的误差

### 2.3 对偶问题

通过拉格朗日对偶性，可以将原始问题转换为对偶问题：

$$ \min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \epsilon \sum_{i=1}^{n} (\alpha_i + \alpha_i^*) - \sum_{i=1}^{n} y_i (\alpha_i - \alpha_i^*) $$

约束条件：
$$ \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) = 0 $$
$$ 0 \leq \alpha_i, \alpha_i^* \leq C $$

其中 $Q_{ij} = K(x_i, x_j)$ 是核矩阵。

### 2.4 预测函数

SVR的预测函数为：

$$ f(x) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(x_i, x) + b $$

#### 2.4.1 支持向量
只有满足以下条件的数据点才会成为支持向量：
- $\alpha_i > 0$ 或 $\alpha_i^* > 0$
- 这些点位于ε-间隔带的边缘或外部

## 第三章：核函数

### 3.1 什么是核函数

核函数是一个函数 $K(x, x')$，它计算两个数据点在高维特征空间中的内积：

$$ K(x, x') = \phi(x)^T \phi(x') $$

其中 $\phi(x)$ 是将原始特征映射到高维特征空间的函数。

### 3.2 常用核函数

#### 3.2.1 线性核
$$ K(x_i, x_j) = x_i^T x_j $$

适用于线性关系的数据，计算速度快。

#### 3.2.2 多项式核
$$ K(x_i, x_j) = (\gamma x_i^T x_j + r)^d $$

其中：
- $\gamma$ 是核系数
- $r$ 是常数项
- $d$ 是多项式次数

适用于多项式关系的数据。

#### 3.2.3 RBF核（径向基函数）
$$ K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) $$

最常用的核函数，适用于大多数情况，可以建模复杂的非线性关系。

#### 3.2.4 Sigmoid核
$$ K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r) $$

类似于神经网络的激活函数，在某些情况下表现良好。

## 第四章：参数调优

### 4.1 C参数

C参数控制正则化强度：

- **C较大**：对误差的惩罚较重，模型更复杂，可能过拟合
- **C较小**：对误差的惩罚较轻，模型更简单，可能欠拟合

#### 4.1.1 选择C的方法
- 网格搜索
- 随机搜索
- 交叉验证

### 4.2 ε参数

ε参数控制ε-间隔带的宽度：

- **ε较大**：允许的误差较大，模型更简单
- **ε较小**：允许的误差较小，模型更复杂

### 4.3 γ参数（RBF核）

γ参数控制RBF核的宽度：

- **γ较大**：核函数较窄，每个数据点的影响范围较小，模型更复杂
- **γ较小**：核函数较宽，每个数据点的影响范围较大，模型更简单

#### 4.3.1 γ的经验规则
- `gamma='scale'`：使用 $1 / (n_{features} * X.var())$ 作为γ值
- `gamma='auto'`：使用 $1 / n_{features}$ 作为γ值

## 第五章：数据预处理

### 5.1 特征标准化

SVR对特征尺度非常敏感，因此需要进行特征标准化：

```python
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
```

### 5.2 为什么需要标准化

- **不同特征的尺度不同**：例如，一个特征的范围是0-1，另一个特征的范围是0-1000
- **核函数对尺度敏感**：RBF核基于距离计算，尺度会影响距离
- **优化过程更稳定**：标准化后优化过程更容易收敛

## 第六章：SVR实现

### 6.1 使用scikit-learn

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 创建SVR模型
svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')

# 训练
svr.fit(X_scaled, y_scaled)

# 预测
y_pred_scaled = svr.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
```

### 6.2 超参数调优

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

## 第七章：SVR的优缺点

### 7.1 优点

1. **非线性建模能力强**：通过核函数可以建模复杂的非线性关系
2. **小样本学习好**：在样本数量较少时表现良好
3. **高维数据稳定**：在特征数量较多时表现稳定
4. **泛化能力强**：通过正则化可以有效防止过拟合
5. **理论基础扎实**：有坚实的统计学习理论基础

### 7.2 缺点

1. **对特征尺度敏感**：需要进行特征标准化
2. **训练复杂度高**：时间复杂度为O(n²)，不适合大规模数据
3. **参数调优复杂**：需要调优C、ε、γ等多个参数
4. **可解释性差**：模型结果难以解释
5. **预测速度慢**：预测时需要计算所有支持向量

## 第八章：实际应用

### 8.1 房价预测

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

### 8.2 时间序列预测

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

## 第九章：总结

### 9.1 核心要点

1. **SVR**是SVM在回归问题上的应用
2. **通过ε-间隔带**允许预测值与真实值之间有一定的误差
3. **使用核技巧**可以建模复杂的非线性关系
4. **需要特征标准化**对特征尺度敏感
5. **参数调优重要**需要仔细调优C、ε、γ等参数
6. **小样本表现好**在样本数量较少时表现良好

### 9.2 学习路径

1. **基础阶段**：理解SVM和SVR的基本原理
2. **实践阶段**：使用Python实现SVR
3. **进阶阶段**：学习核函数和参数调优
4. **应用阶段**：在实际项目中应用SVR

---

**练习题目**：

1. 推导SVR的对偶问题。
2. 实现一个简单的线性SVR算法。
3. 比较不同核函数在SVR中的效果。
4. 使用网格搜索调优SVR的参数。
5. 使用SVR对一个时间序列进行预测。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Nature of Statistical Learning Theory》Vapnik
- 《Pattern Recognition and Machine Learning》Bishop
- 《Support Vector Regression》Drucker et al.
