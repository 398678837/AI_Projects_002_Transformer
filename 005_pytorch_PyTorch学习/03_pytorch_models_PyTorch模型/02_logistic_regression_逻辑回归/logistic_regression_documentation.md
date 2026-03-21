# PyTorch 逻辑回归详细文档

## 1. 逻辑回归的基本概念

逻辑回归是一种用于分类问题的监督学习算法，它通过将线性回归的输出通过sigmoid函数映射到[0,1]区间，从而预测样本属于某个类别的概率。

### 1.1 数学表达式

对于二分类问题，逻辑回归的模型可以表示为：

 P(y=1|x) = sigmoid(w^T x + b) = 1 / (1 + e^{-(w^T x + b)}) 

其中，w是权重向量，b是偏置，sigmoid函数将线性组合的结果映射到[0,1]区间。

对于多分类问题，逻辑回归通常使用softmax函数：

 P(y=k|x) = e^{w_k^T x + b_k} / sum_{i=1}^K e^{w_i^T x + b_i} 

其中，K是类别数，w_k是第k类的权重向量，b_k是第k类的偏置。

## 2. PyTorch 实现逻辑回归

### 2.1 基本逻辑回归

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 创建模型、损失函数和优化器
model = LogisticRegression(X.shape[1])
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
    print(f'测试准确率: {accuracy:.4f}')
```

### 2.2 多分类逻辑回归

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成多分类数据
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, n_clusters_per_class=1, n_redundant=0, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)  # 多分类任务中，标签不需要one-hot编码

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
class MultiClassLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)  # 多分类任务中，不需要sigmoid，使用CrossEntropyLoss会自动应用softmax

# 创建模型、损失函数和优化器
model = MultiClassLogisticRegression(X.shape[1], 3)
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, y_pred_class = torch.max(y_pred, 1)
    accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
    print(f'测试准确率: {accuracy:.4f}')
```

## 3. 损失函数

### 3.1 二分类交叉熵损失 (BCELoss)

二分类交叉熵损失用于二分类问题，计算预测概率与真实标签之间的交叉熵：

 BCELoss = - (y log(p) + (1 - y) log(1 - p)) 

在PyTorch中，可以使用`nn.BCELoss()`实现：

```python
criterion = nn.BCELoss()
loss = criterion(outputs, y)
```

### 3.2 多分类交叉熵损失 (CrossEntropyLoss)

多分类交叉熵损失用于多分类问题，计算预测概率分布与真实标签之间的交叉熵：

 CrossEntropyLoss = - sum_{k=1}^K y_k log(p_k) 

在PyTorch中，可以使用`nn.CrossEntropyLoss()`实现：

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(outputs, y)
```

### 3.3 带权重的交叉熵损失

当数据集不平衡时，可以使用带权重的交叉熵损失：

```python
# 计算类别权重
class_weights = torch.tensor([1.0, 2.0])  # 假设第二类样本较少，给予更高的权重
criterion = nn.BCELoss(weight=class_weights)
```

## 4. 优化器

### 4.1 随机梯度下降 (SGD)

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 4.2 动量优化器

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 4.3 Adam 优化器

```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

## 5. 正则化

### 5.1 L2 正则化

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
```

### 5.2 L1 正则化

```python
# 手动实现L1正则化
l1_lambda = 0.01
loss = criterion(outputs, y) + l1_lambda * sum(p.abs().sum() for p in model.parameters())
```

## 6. 模型评估

### 6.1 准确率

```python
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()  # 二分类
    # y_pred_class = torch.argmax(y_pred, dim=1)  # 多分类
    accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
    print(f'测试准确率: {accuracy:.4f}')
```

### 6.2 混淆矩阵

```python
from sklearn.metrics import confusion_matrix

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    cm = confusion_matrix(y_test.numpy(), y_pred_class.numpy())
    print('混淆矩阵:')
    print(cm)
```

### 6.3 精确率、召回率和F1分数

```python
from sklearn.metrics import precision_recall_fscore_support

with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    precision, recall, f1, _ = precision_recall_fscore_support(y_test.numpy(), y_pred_class.numpy(), average='binary')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    print(f'F1分数: {f1:.4f}')
```

## 7. 实际应用

### 7.1 数据预处理

```python
from sklearn.preprocessing import StandardScaler

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = torch.tensor(X_scaled, dtype=torch.float32)
```

### 7.2 交叉验证

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 训练模型
    model = LogisticRegression(X.shape[1])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练过程...
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
    
    print(f'Fold {fold+1}, 准确率: {accuracy:.4f}')
```

### 7.3 超参数调优

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

# 使用scikit-learn的GridSearchCV进行超参数调优
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SKLogisticRegression())
])

param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X.numpy(), y.numpy().ravel())

print(f'最佳参数: {grid_search.best_params_}')
print(f'最佳准确率: {grid_search.best_score_:.4f}')
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

### 8.3 类别不平衡

**问题**：数据集类别分布不平衡，导致模型偏向于多数类

**解决方案**：
- 数据重采样（过采样少数类或欠采样多数类）
- 使用带权重的损失函数
- 使用F1分数等更合适的评估指标

### 8.4 特征选择

**问题**：特征过多，导致模型过拟合或训练缓慢

**解决方案**：
- 使用L1正则化进行特征选择
- 使用递归特征消除（RFE）
- 使用相关性分析选择重要特征

## 9. 代码优化技巧

### 9.1 批量训练

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

```python
# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 将模型和数据移到GPU
model.to(device)
X = X.to(device)
y = y.to(device)
```

### 9.3 模型保存和加载

```python
# 保存模型
torch.save(model.state_dict(), 'logistic_regression_model.pth')

# 加载模型
model = LogisticRegression(input_size)
model.load_state_dict(torch.load('logistic_regression_model.pth'))
model.eval()
```

## 10. 总结

逻辑回归是一种简单但强大的分类算法，适用于二分类和多分类问题。本文档介绍了如何使用PyTorch实现逻辑回归，包括基本逻辑回归、多分类逻辑回归，以及损失函数、优化器、正则化等相关概念。

**主要内容**：

1. **逻辑回归的基本概念**：二分类和多分类逻辑回归的数学表达式
2. **PyTorch实现**：使用PyTorch构建和训练逻辑回归模型
3. **损失函数**：BCELoss、CrossEntropyLoss等损失函数的使用
4. **优化器**：SGD、动量优化器、Adam等优化器的使用
5. **正则化**：L1和L2正则化的实现
6. **模型评估**：准确率、混淆矩阵、精确率、召回率和F1分数
7. **实际应用**：数据预处理、交叉验证、超参数调优
8. **常见问题与解决方案**：过拟合、欠拟合、类别不平衡、特征选择
9. **代码优化技巧**：批量训练、GPU加速、模型保存和加载

通过本文档的学习，您应该已经掌握了使用PyTorch实现逻辑回归的基本方法，可以开始应用到实际问题中了。