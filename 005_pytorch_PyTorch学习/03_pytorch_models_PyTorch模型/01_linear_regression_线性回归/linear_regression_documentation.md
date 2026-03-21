# PyTorch 线性回归详细文档

## 1. 线性回归的基本概念

线性回归是一种用于建模自变量和因变量之间线性关系的统计方法。在机器学习中，线性回归是一种监督学习算法，用于预测连续值输出。

### 1.1 数学表达式

对于单变量线性回归，模型可以表示为：

 y = wx + b 

其中，w是权重，b是偏置。

对于多变量线性回归，模型可以表示为：

 y = w_1x_1 + w_2x_2 + ... + w_nx_n + b 

或者用向量形式表示：

 y = W^T X + b 

其中，W是权重向量，X是输入特征向量。

## 2. PyTorch 实现线性回归

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

## 3. 损失函数

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

## 4. 优化器

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

## 5. 正则化

### 5.1 L2 正则化

L2正则化（也称为权重衰减）通过在损失函数中添加权重的平方和来防止过拟合：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
```

### 5.2 L1 正则化

L1正则化通过在损失函数中添加权重的绝对值和来产生稀疏权重：

```python
# 手动实现L1正则化
l1_lambda = 0.01
loss = criterion(outputs, y) + l1_lambda * sum(p.abs().sum() for p in model.parameters())
```

## 6. 模型评估

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

## 7. 实际应用

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

### 7.2 交叉验证

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

### 7.3 超参数调优

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

## 8. 常见问题与解决方案

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

## 9. 代码优化技巧

### 9.1 批量训练

对于大型数据集，使用批量训练可以提高训练速度：

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 9.2 GPU加速

使用GPU可以显著提高训练速度：

```python
# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 将模型和数据移到GPU
model.to(device)
x = x.to(device)
y = y.to(device)
```

### 9.3 模型保存和加载

保存和加载模型可以避免重复训练：

```python
# 保存模型
torch.save(model.state_dict(), 'linear_regression_model.pth')

# 加载模型
model = LinearRegression()
model.load_state_dict(torch.load('linear_regression_model.pth'))
model.eval()
```

## 10. 总结

线性回归是一种简单但强大的机器学习算法，适用于预测连续值输出。本文档介绍了如何使用PyTorch实现线性回归，包括基本线性回归、多元线性回归和多项式回归，以及损失函数、优化器、正则化等相关概念。

**主要内容**：

1. **线性回归的基本概念**：单变量和多变量线性回归的数学表达式
2. **PyTorch实现**：使用PyTorch构建和训练线性回归模型
3. **损失函数**：MSE、MAE、RMSE等损失函数的使用
4. **优化器**：SGD、动量优化器、Adam等优化器的使用
5. **正则化**：L1和L2正则化的实现
6. **模型评估**：训练和测试损失的监控，R²评分等评估指标
7. **实际应用**：数据预处理、交叉验证、超参数调优
8. **常见问题与解决方案**：过拟合、欠拟合、学习率选择、特征缩放
9. **代码优化技巧**：批量训练、GPU加速、模型保存和加载

通过本文档的学习，您应该已经掌握了使用PyTorch实现线性回归的基本方法，可以开始应用到实际问题中了。