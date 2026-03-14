# 神经网络回归（Neural Network Regression）详细文档

## 1. 概念介绍

### 1.1 什么是神经网络回归
神经网络回归是使用人工神经网络（Artificial Neural Network, ANN）来解决回归问题的方法。神经网络通过多层非线性变换来学习输入特征与目标变量之间的复杂非线性关系。

### 1.2 核心思想
- **多层感知机（MLP）**：由输入层、隐藏层和输出层组成
- **非线性激活函数**：使用ReLU、Sigmoid、Tanh等激活函数引入非线性
- **反向传播**：通过反向传播算法训练网络
- **梯度下降**：使用梯度下降优化损失函数

### 1.3 应用场景
- **复杂非线性关系**：当特征与目标变量之间存在复杂的非线性关系时
- **大规模数据**：在数据量很大时表现良好
- **图像回归**：从图像中预测连续值
- **时间序列预测**：某些复杂时间序列的预测
- **多任务学习**：同时预测多个目标变量

## 2. 技术原理

### 2.1 神经网络结构

#### 2.1.1 多层感知机（MLP）
MLP是最常用的神经网络结构：
- **输入层**：接收输入特征
- **隐藏层**：进行非线性变换
- **输出层**：输出预测值（回归问题通常只有一个节点）

#### 2.1.2 前向传播
$$ z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)} $$
$$ a^{(l)} = g^{(l)}(z^{(l)}) $$

其中：
- $W^{(l)}$ 是第l层的权重矩阵
- $b^{(l)}$ 是第l层的偏置
- $g^{(l)}$ 是第l层的激活函数

### 2.2 激活函数

#### 2.2.1 ReLU（Rectified Linear Unit）
$$ g(z) = \max(0, z) $$
- 最常用的激活函数
- 计算简单
- 缓解梯度消失问题

#### 2.2.2 Sigmoid
$$ g(z) = \frac{1}{1 + e^{-z}} $$
- 输出在(0, 1)之间
- 容易产生梯度消失

#### 2.2.3 Tanh
$$ g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} $$
- 输出在(-1, 1)之间
- 比Sigmoid的梯度大

#### 2.2.4 恒等（Identity）
$$ g(z) = z $$
- 通常用于输出层（回归问题）

### 2.3 损失函数

对于回归问题，通常使用均方误差（MSE）：

$$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

### 2.4 反向传播

反向传播算法通过链式法则计算梯度：

$$ \frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} a^{(l-1)T} $$
$$ \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)} $$

其中 $\delta^{(l)}$ 是第l层的误差项。

## 3. 代码实现

### 3.1 scikit-learn实现

文件：`nn_regression_demo.py`

#### 3.1.1 核心步骤
1. **数据准备**：加载或生成数据
2. **数据标准化**：神经网络对特征尺度敏感，需要标准化
3. **模型创建**：使用MLPRegressor类
4. **模型训练**：训练神经网络回归模型
5. **模型评估**：计算MSE、RMSE和R²评分
6. **可视化**：分析训练曲线和预测结果

#### 3.1.2 关键代码

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 标准化数据
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 创建神经网络回归模型
nn = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # 隐藏层结构
    activation='relu',               # 激活函数
    solver='adam',                   # 优化器
    max_iter=1000,                   # 最大迭代次数
    random_state=42,                 # 随机种子
    early_stopping=True,             # 早停
    validation_fraction=0.2          # 验证集比例
)

# 训练模型
nn.fit(X_scaled, y_scaled)

# 预测
y_pred_scaled = nn.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 获取训练损失
loss_curve = nn.loss_curve_
```

## 4. 超参数调优

### 4.1 重要参数

#### 4.1.1 网络结构
- **hidden_layer_sizes**：隐藏层的节点数，如(100,)、(100, 50)
- **层数**：隐藏层的数量

#### 4.1.2 训练参数
- **activation**：激活函数（'identity'、'logistic'、'tanh'、'relu'）
- **solver**：优化器（'lbfgs'、'sgd'、'adam'）
- **alpha**：L2正则化参数
- **learning_rate**：学习率（'constant'、'invscaling'、'adaptive'）
- **max_iter**：最大迭代次数
- **early_stopping**：是否使用早停
- **validation_fraction**：验证集比例

### 4.2 网格搜索

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}

grid_search = GridSearchCV(
    MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_scaled, y_scaled)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证R²: {grid_search.best_score_:.4f}")
```

## 5. 优缺点分析

### 5.1 优点

- **非线性建模能力强**：可以学习复杂的非线性关系
- **自动特征学习**：可以自动学习特征表示
- **端到端学习**：可以直接从原始数据学习
- **扩展性好**：可以通过增加网络层数和节点数来提高性能
- **适用范围广**：适用于各种类型的数据

### 5.2 缺点

- **需要大量数据**：在小样本上容易过拟合
- **计算资源需求大**：训练需要大量计算资源
- **参数调优复杂**：需要调优很多超参数
- **可解释性差**：黑盒模型，难以解释
- **对特征尺度敏感**：需要进行特征标准化
- **容易过拟合**：需要使用正则化和早停

## 6. 实际应用

### 6.1 房价预测

```python
from sklearn.datasets import fetch_california_housing
from sklearn.neural_network import MLPRegressor
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
nn = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
nn.fit(X_scaled, y_scaled)
```

---

**参考资料**：
- 《Deep Learning》Goodfellow, Bengio, Courville
- scikit-learn官方文档
- 《Neural Networks and Deep Learning》Michael Nielsen
- 《Pattern Recognition and Machine Learning》Bishop
