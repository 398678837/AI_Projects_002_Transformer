# PyTorch 线性回归教材

## 第一章：线性回归的基本概念

### 1.1 什么是线性回归

线性回归是一种用于建模自变量和因变量之间线性关系的统计方法。在机器学习中，线性回归是一种监督学习算法，用于预测连续值输出。

### 1.2 线性回归的应用场景

- **房价预测**：根据房屋面积、位置等特征预测房价
- **销售预测**：根据历史销售数据预测未来销售额
- **股票价格预测**：根据历史价格和市场指标预测股票价格
- **学生成绩预测**：根据学习时间、出勤率等预测学生成绩

### 1.3 线性回归的数学表达式

对于单变量线性回归，模型可以表示为：

 y = wx + b 

其中，w是权重，b是偏置。

对于多变量线性回归，模型可以表示为：

 y = w_1x_1 + w_2x_2 + ... + w_nx_n + b 

或者用向量形式表示：

 y = W^T X + b 

其中，W是权重向量，X是输入特征向量。

## 第二章：PyTorch 实现线性回归

### 2.1 基本线性回归

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = torch.linspace(0, 10, 100).unsqueeze(1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.5

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型、损失函数和优化器
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    predicted = model(x)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='实际数据')
plt.plot(x.numpy(), predicted.numpy(), 'r-', label='预测结果')
plt.title('线性回归')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 多元线性回归

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 生成数据
n_samples = 100
n_features = 3
x = torch.randn(n_samples, n_features)
# 真实参数
weights = torch.tensor([2.5, -1.5, 0.8])
bias = 1.2
y = x @ weights + bias + torch.randn(n_samples) * 0.5
y = y.unsqueeze(1)

# 定义模型
class MultipleLinearRegression(nn.Module):
    def __init__(self, input_size):
        super(MultipleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型、损失函数和优化器
model = MultipleLinearRegression(n_features)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印真实参数和预测参数
print("\n真实参数:")
print(f'权重: {weights}')
print(f'偏置: {bias}')
print("\n预测参数:")
print(f'权重: {model.linear.weight.data.numpy()[0]}')
print(f'偏置: {model.linear.bias.data.numpy()[0]}')
```

### 2.3 多项式回归

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据
x = torch.linspace(-3, 3, 100).unsqueeze(1)
y = x**3 + 2*x**2 - 3*x + 1 + torch.randn(100, 1) * 2

# 特征转换（添加多项式特征）
def polynomial_features(x, degree):
    features = []
    for i in range(1, degree+1):
        features.append(x**i)
    return torch.cat(features, dim=1)

degree = 3
x_poly = polynomial_features(x, degree)

# 定义模型
class PolynomialRegression(nn.Module):
    def __init__(self, input_size):
        super(PolynomialRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型、损失函数和优化器
model = PolynomialRegression(degree)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_poly)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    predicted = model(x_poly)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label='实际数据')
plt.plot(x.numpy(), predicted.numpy(), 'r-', label='预测结果')
plt.title(f'{degree}阶多项式回归')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

## 第三章：损失函数

### 3.1 均方误差 (MSE)

均方误差是线性回归中最常用的损失函数，计算预测值与真实值之间差异的平方和的平均值：

 MSE = (1/n) sum_{i=1}^n (y_i - hat{y}_i)^2 

在PyTorch中，可以使用`nn.MSELoss()`实现：

```python
criterion = nn.MSELoss()
loss = criterion(outputs, y)
```

### 3.2 平均绝对误差 (MAE)

平均绝对误差计算预测值与真实值之间差异的绝对值的平均值：

 MAE = (1/n) sum_{i=1}^n |y_i - hat{y}_i| 

在PyTorch中，可以使用`nn.L1Loss()`实现：

```python
criterion = nn.L1Loss()
loss = criterion(outputs, y)
```

### 3.3 均方根误差 (RMSE)

均方根误差是均方误差的平方根：

 RMSE = sqrt((1/n) sum_{i=1}^n (y_i - hat{y}_i)^2) 

在PyTorch中，可以通过对MSE取平方根实现：

```python
criterion = nn.MSELoss()
loss = torch.sqrt(criterion(outputs, y))
```

### 3.4 损失函数的选择

- **MSE**：对 outliers敏感，适用于数据分布较为均匀的情况
- **MAE**：对 outliers不敏感，适用于数据中存在较多异常值的情况
- **RMSE**：与目标变量具有相同的单位，更直观

## 第四章：优化器

### 4.1 随机梯度下降 (SGD)

随机梯度下降是最基本的优化算法，通过计算参数的梯度并沿梯度负方向更新参数：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 4.2 动量优化器

动量优化器在SGD的基础上添加了动量项，有助于加速收敛：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 4.3 Adam 优化器

Adam优化器结合了动量优化和自适应学习率，通常能够更快地收敛：

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### 4.4 优化器的选择

- **SGD**：适用于简单模型和较小的数据集
- **动量优化器**：适用于需要加速收敛的情况
- **Adam**：适用于复杂模型和较大的数据集，通常是默认选择

## 第五章：正则化

### 5.1 过拟合问题

过拟合是指模型在训练集上表现良好，但在测试集上表现差的现象。过拟合通常发生在模型过于复杂或训练数据不足的情况下。

### 5.2 L2 正则化

L2正则化（也称为权重衰减）通过在损失函数中添加权重的平方和来防止过拟合：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
```

### 5.3 L1 正则化

L1正则化通过在损失函数中添加权重的绝对值和来产生稀疏权重：

```python
# 手动实现L1正则化
l1_lambda = 0.01
loss = criterion(outputs, y) + l1_lambda * sum(p.abs().sum() for p in model.parameters())
```

### 5.4 正则化强度的选择

- 正则化强度过大：可能导致欠拟合
- 正则化强度过小：可能无法有效防止过拟合
- 需要通过交叉验证来选择合适的正则化强度

## 第六章：模型评估

### 6.1 训练和测试损失

在训练过程中，应该同时监控训练损失和测试损失，以防止过拟合：

```python
# 训练
model.train()
outputs = model(X_train)
loss = criterion(outputs, y_train)

# 测试
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
```

### 6.2 评估指标

除了损失函数外，还可以使用其他评估指标来评估模型性能：

- **R² 评分**：衡量模型解释因变量变异的比例
- **调整R²**：考虑特征数量的R²评分

```python
from sklearn.metrics import r2_score

with torch.no_grad():
    y_pred = model(X_test).numpy()
    r2 = r2_score(y_test.numpy(), y_pred)
    print(f'R² 评分: {r2:.4f}')
```

### 6.3 交叉验证

交叉验证是一种评估模型性能的方法，通过将数据分成多个折叠，使用不同的折叠作为测试集：

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 训练模型
    model = LinearRegression(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练过程...
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
    
    print(f'Fold {fold+1}, Test Loss: {test_loss.item():.4f}')
```

## 第七章：实际应用

### 7.1 数据预处理

在实际应用中，通常需要对数据进行预处理：

- **特征标准化**：将特征缩放到均值为0，标准差为1
- **特征选择**：选择与目标变量相关的特征
- **处理缺失值**：填充或删除缺失值

```python
from sklearn.preprocessing import StandardScaler

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = torch.tensor(X_scaled, dtype=torch.float32)
```

### 7.2 超参数调优

超参数调优是提高模型性能的重要步骤：

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression as SKLinearRegression

# 使用scikit-learn的GridSearchCV进行超参数调优
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', SKLinearRegression())
])

param_grid = {
    'regressor__fit_intercept': [True, False]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X.numpy(), y.numpy().ravel())

print(f'最佳参数: {grid_search.best_params_}')
print(f'最佳R² 评分: {grid_search.best_score_:.4f}')
```

### 7.3 模型部署

训练好的模型可以部署到生产环境中：

```python
# 保存模型
torch.save(model.state_dict(), 'linear_regression_model.pth')

# 加载模型
model = LinearRegression()
model.load_state_dict(torch.load('linear_regression_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    new_data = torch.tensor([[1.0, 2.0, 3.0]])
    prediction = model(new_data)
    print(f'预测结果: {prediction.item()}')
```

## 第八章：常见问题与解决方案

### 8.1 过拟合

**问题**：模型在训练集上表现良好，但在测试集上表现差

**解决方案**：
- 增加训练数据
- 使用正则化
- 减少模型复杂度
- 早期停止

### 8.2 欠拟合

**问题**：模型在训练集和测试集上表现都差

**解决方案**：
- 增加模型复杂度
- 添加更多特征
- 减少正则化强度

### 8.3 学习率选择

**问题**：学习率过大导致训练不稳定，学习率过小导致收敛缓慢

**解决方案**：
- 使用学习率调度器
- 尝试不同的学习率值
- 使用自适应学习率优化器（如Adam）

### 8.4 特征缩放

**问题**：特征之间的尺度差异导致训练困难

**解决方案**：
- 使用StandardScaler进行特征标准化
- 使用MinMaxScaler进行特征缩放到[0,1]范围

### 8.5 梯度消失/爆炸

**问题**：训练过程中梯度变得非常小或非常大

**解决方案**：
- 使用合适的初始化方法
- 梯度裁剪
- 使用批量归一化

## 第九章：习题

### 9.1 选择题

1. 线性回归的目标是：
   A. 最小化预测值与真实值之间的差异
   B. 最大化预测值与真实值之间的差异
   C. 最小化模型参数
   D. 最大化模型参数

2. 以下哪个是线性回归中常用的损失函数？
   A. 交叉熵损失
   B. 均方误差
   C. 绝对误差
   D.  hinge损失

3. L2正则化的作用是：
   A. 增加模型复杂度
   B. 防止过拟合
   C. 加速训练
   D. 提高模型精度

### 9.2 填空题

1. 单变量线性回归的数学表达式是__________。
2. 多元线性回归中，输入特征的数量称为__________。
3. 过拟合是指模型在__________上表现良好，但在__________上表现差。

### 9.3 简答题

1. 解释线性回归的基本原理。
2. 比较MSE和MAE损失函数的区别。
3. 什么是正则化？它的作用是什么？

### 9.4 编程题

1. 实现一个单变量线性回归模型，使用PyTorch训练并可视化结果。

2. 实现一个多元线性回归模型，使用PyTorch训练并评估模型性能。

3. 实现一个多项式回归模型，尝试不同的多项式阶数，观察模型性能的变化。

4. 使用真实数据集（如波士顿房价数据集）训练线性回归模型，并使用交叉验证评估模型性能。

## 第十章：总结

### 10.1 知识回顾

1. **线性回归的基本概念**：单变量和多变量线性回归的数学表达式
2. **PyTorch实现**：使用PyTorch构建和训练线性回归模型
3. **损失函数**：MSE、MAE、RMSE等损失函数的使用
4. **优化器**：SGD、动量优化器、Adam等优化器的使用
5. **正则化**：L1和L2正则化的实现
6. **模型评估**：训练和测试损失的监控，R²评分等评估指标
7. **实际应用**：数据预处理、交叉验证、超参数调优
8. **常见问题与解决方案**：过拟合、欠拟合、学习率选择、特征缩放

### 10.2 学习建议

1. **实践练习**：尝试使用不同的数据集和参数设置来训练线性回归模型
2. **理解原理**：深入理解线性回归的数学原理和PyTorch的实现细节
3. **模型调优**：学习如何通过调整超参数来提高模型性能
4. **扩展学习**：学习其他回归模型，如岭回归、LASSO回归等
5. **实际应用**：尝试将线性回归应用到实际问题中

### 10.3 进阶学习

1. **岭回归**：带有L2正则化的线性回归
2. **LASSO回归**：带有L1正则化的线性回归
3. **弹性网络**：结合L1和L2正则化的线性回归
4. **多项式回归**：通过特征转换实现的非线性回归
5. **支持向量回归**：使用支持向量机进行回归
6. **决策树回归**：使用决策树进行回归
7. **集成方法**：随机森林、梯度提升树等集成方法

通过本章的学习，您应该已经掌握了线性回归的基本原理和PyTorch实现方法，可以开始应用到实际问题中了。