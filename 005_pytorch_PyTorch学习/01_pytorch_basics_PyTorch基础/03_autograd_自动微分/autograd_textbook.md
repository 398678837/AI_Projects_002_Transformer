# PyTorch 自动微分教材

## 第一章：自动微分的基本概念

### 1.1 什么是自动微分

自动微分（Automatic Differentiation）是一种计算函数导数的技术，它可以自动计算复杂函数的梯度。在深度学习中，自动微分是训练模型的核心，因为它允许我们计算损失函数对模型参数的梯度，从而使用梯度下降算法来优化模型。

### 1.2 计算图

PyTorch使用动态计算图来跟踪张量的操作。计算图是一种有向无环图（DAG），其中节点表示张量，边表示张量之间的操作。当我们对张量进行操作时，PyTorch会自动构建计算图，记录所有的操作步骤。

### 1.3 正向传播和反向传播

- **正向传播**：从输入张量开始，按照计算图的顺序执行操作，计算输出值。
- **反向传播**：从输出值开始，按照计算图的反向顺序计算梯度，使用链式法则自动计算每个参数的梯度。

## 第二章：基本自动微分

### 2.1 创建需要梯度的张量

要使用自动微分，首先需要创建一个设置了`requires_grad=True`的张量。

```python
import torch

# 创建一个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
print("x:", x)
print("x.requires_grad:", x.requires_grad)
```

### 2.2 计算梯度

定义一个函数，然后调用`backward()`方法计算梯度。

```python
# 定义一个函数 y = x^2
y = x ** 2
print("y:", y)

# 计算梯度
y.backward()
print("x.grad:", x.grad)  # 应该是 4.0
```

### 2.3 清除梯度

梯度会累积，所以在每次计算梯度前，需要清除之前的梯度。

```python
# 清除梯度
x.grad.zero_()
print("清除梯度后 x.grad:", x.grad)
```

## 第三章：多变量自动微分

### 3.1 多个输入变量

当函数有多个输入变量时，PyTorch会为每个输入变量计算梯度。

```python
# 创建两个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义一个函数 z = x^2 + y^2
z = x ** 2 + y ** 2
print("z:", z)

# 计算梯度
z.backward()
print("x.grad:", x.grad)  # 应该是 4.0
print("y.grad:", y.grad)  # 应该是 6.0
```

### 3.2 链式法则

当计算图比较复杂时，PyTorch会使用链式法则自动计算梯度。

```python
# 创建输入张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义复杂计算
z = x ** 2 * y + y
print("z:", z)

# 计算梯度
z.backward()
print("x.grad:", x.grad)  # 应该是 2 * x * y = 2 * 2 * 3 = 12
print("y.grad:", y.grad)  # 应该是 x^2 + 1 = 4 + 1 = 5
```

## 第四章：向量和矩阵的自动微分

### 4.1 向量的自动微分

对于向量输入，PyTorch会计算向量的梯度。

```python
# 创建向量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print("x:", x)

# 计算向量的范数
y = torch.norm(x)
print("y:", y)

# 计算梯度
y.backward()
print("x.grad:", x.grad)  # 应该是 x / ||x||
```

### 4.2 矩阵的自动微分

对于矩阵输入，PyTorch会计算矩阵的梯度。

```python
# 矩阵的自动微分
A = torch.randn(2, 3, requires_grad=True)
B = torch.randn(3, 2, requires_grad=True)
C = A @ B
print("C:", C)

# 计算梯度
C.sum().backward()
print("A.grad:", A.grad)
print("B.grad:", B.grad)
```

## 第五章：梯度累积

### 5.1 梯度累积的现象

当多次调用`backward()`方法时，梯度会累积。

```python
x = torch.tensor(1.0, requires_grad=True)

# 第一次前向传播
y = x ** 2
y.backward()
print("第一次 backward 后 x.grad:", x.grad)  # 2

# 第二次前向传播（不清除梯度）
y = x ** 2
y.backward()
print("第二次 backward 后 x.grad:", x.grad)  # 4（梯度累积）
```

### 5.2 清除梯度

使用`zero_()`方法清除梯度，或者使用优化器的`zero_grad()`方法。

```python
# 清除梯度后再计算
x.grad.zero_()
y = x ** 2
y.backward()
print("清除梯度后 x.grad:", x.grad)  # 2
```

## 第六章：禁用梯度计算

### 6.1 使用`torch.no_grad()`

在某些情况下，我们可能不需要计算梯度，例如在推理阶段。

```python
x = torch.tensor(1.0, requires_grad=True)

# 正常计算梯度
y = x ** 2
y.backward()
print("正常计算后 x.grad:", x.grad)

# 禁用梯度计算
x.grad.zero_()
with torch.no_grad():
    y = x ** 2
    print("禁用梯度计算时 y.requires_grad:", y.requires_grad)
```

### 6.2 使用`detach()`

`detach()`方法可以创建一个与原张量共享数据但不参与梯度计算的新张量。

```python
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
z = y.detach()  # 创建一个不参与梯度计算的张量
print("z.requires_grad:", z.requires_grad)
```

## 第七章：自定义函数的自动微分

### 7.1 基本自定义函数

对于简单的自定义函数，PyTorch可以自动计算梯度。

```python
# 定义一个自定义函数
def f(x):
    return x ** 3 + 2 * x

x = torch.tensor(2.0, requires_grad=True)
y = f(x)
print("f(2):", y)

# 计算梯度
y.backward()
print("f'(2):", x.grad)  # 应该是 3x^2 + 2 = 3*4 + 2 = 14
```

### 7.2 自定义自动微分函数

对于复杂的自定义函数，可以使用`torch.autograd.Function`来定义自动微分。

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播
        result = x ** 2
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播
        x, = ctx.saved_tensors
        grad_input = 2 * x * grad_output
        return grad_input

# 使用自定义函数
x = torch.tensor(2.0, requires_grad=True)
y = MyFunction.apply(x)
y.backward()
print("x.grad:", x.grad)
```

## 第八章：线性回归中的自动微分

### 8.1 简单线性回归

```python
# 生成数据
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1) * 0.1

# 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# 训练参数
learning_rate = 0.01
num_epochs = 100

for epoch in range(num_epochs):
    # 前向传播
    y_pred = X @ w + b
    
    # 计算损失
    loss = torch.mean((y_pred - y) ** 2)
    
    # 计算梯度
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 清除梯度
    w.grad.zero_()
    b.grad.zero_()

print("训练完成")
print("w:", w.item())
print("b:", b.item())
```

## 第九章：神经网络中的自动微分

### 9.1 简单神经网络

```python
# 生成数据
X = torch.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

# 定义神经网络
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.fc2 = torch.nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 初始化模型
model = NeuralNetwork()

# 定义损失函数和优化器
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = criterion(y_pred, y)
    
    # 计算梯度
    optimizer.zero_grad()
    loss.backward()
    
    # 更新参数
    optimizer.step()

print("训练完成")
```

## 第十章：性能测试

### 10.1 自动微分性能测试

```python
import time

# 测试不同大小张量的自动微分性能
sizes = [100, 1000, 5000]
times = []

for size in sizes:
    # 创建张量
    x = torch.randn(size, size, requires_grad=True)
    
    # 前向传播
    start = time.time()
    y = x.sum()
    
    # 反向传播
    y.backward()
    end = time.time()
    
    times.append(end - start)
    print(f"大小为 {size}x{size} 的张量自动微分耗时: {end - start:.6f} 秒")
```

## 第十一章：常见问题与解决方案

### 11.1 梯度为None

**问题**：调用`backward()`后，张量的`grad`属性为`None`

**解决方案**：
- 确保张量在创建时设置了`requires_grad=True`
- 确保张量参与了计算图的构建
- 确保调用了`backward()`方法

### 11.2 梯度累积

**问题**：多次调用`backward()`后，梯度会累积

**解决方案**：
- 在每次调用`backward()`前，使用`zero_()`方法清除梯度
- 使用优化器的`zero_grad()`方法清除所有参数的梯度

### 11.3 计算图太大

**问题**：计算图太大，导致内存不足

**解决方案**：
- 减小批量大小
- 使用`torch.no_grad()`禁用不需要的梯度计算
- 使用`detach()`方法分离张量，停止跟踪历史

### 11.4 无法计算梯度

**问题**：某些操作无法计算梯度

**解决方案**：
- 检查操作是否支持自动微分
- 使用`torch.autograd.Function`自定义自动微分函数

## 第十二章：高级自动微分技术

### 12.1 高阶导数

PyTorch支持计算高阶导数。

```python
# 计算二阶导数
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

# 一阶导数
y.backward(create_graph=True)
print("一阶导数:", x.grad)

# 二阶导数
x.grad.zero_()
gradient = x.grad
x.grad.backward()
print("二阶导数:", x.grad)
```

### 12.2 梯度检查

使用`torch.autograd.gradcheck`来检查梯度计算是否正确。

```python
from torch.autograd import gradcheck

# 定义一个函数
def f(x):
    return x ** 2

# 创建输入张量
x = torch.tensor(2.0, requires_grad=True)

# 检查梯度
result = gradcheck(f, (x,), eps=1e-6, atol=1e-4)
print("梯度检查结果:", result)
```

## 第十三章：习题

### 13.1 选择题

1. 要计算张量的梯度，应该调用哪个方法？
   A. grad()
   B. backward()
   C. derivative()
   D. diff()

2. 要清除张量的梯度，应该调用哪个方法？
   A. clear()
   B. reset()
   C. zero_()
   D. empty()

3. 要禁用梯度计算，应该使用哪个上下文管理器？
   A. torch.no_grad()
   B. torch.without_grad()
   C. torch.grad_off()
   D. torch.no_gradient()

### 13.2 填空题

1. 要创建一个需要梯度的张量，应该设置 __________ 参数为True。
2. 梯度会 __________，所以在每次计算梯度前需要清除之前的梯度。
3. 对于向量或矩阵输入，调用`backward()`时需要传递 __________。

### 13.3 简答题

1. 解释计算图的概念。
2. 解释正向传播和反向传播的过程。
3. 什么是链式法则？它在自动微分中有什么作用？

### 13.4 编程题

1. 创建一个张量`x = torch.tensor(3.0, requires_grad=True)`，计算函数`y = x^3 - 2x^2 + 5`的梯度。

2. 创建两个张量`x = torch.tensor(2.0, requires_grad=True)`和`y = torch.tensor(3.0, requires_grad=True)`，计算函数`z = x^2 * y + torch.sin(x)`的梯度。

3. 实现一个简单的线性回归模型，使用自动微分来训练模型。

4. 实现一个简单的神经网络，使用自动微分来训练模型。

## 第十四章：总结

### 14.1 知识回顾

1. **自动微分的基本概念**：自动微分是计算函数导数的技术，是深度学习的核心
2. **计算图**：PyTorch使用动态计算图来跟踪张量的操作
3. **基本自动微分**：创建需要梯度的张量，计算梯度，清除梯度
4. **多变量自动微分**：计算多个输入变量的梯度，使用链式法则
5. **向量和矩阵的自动微分**：计算向量和矩阵的梯度
6. **梯度累积**：多次调用`backward()`时梯度会累积
7. **禁用梯度计算**：使用`torch.no_grad()`禁用不需要的梯度计算
8. **自定义函数的自动微分**：使用`torch.autograd.Function`自定义自动微分函数
9. **线性回归中的自动微分**：使用自动微分训练线性回归模型
10. **神经网络中的自动微分**：使用自动微分训练神经网络
11. **性能测试**：测试自动微分的性能
12. **常见问题与解决方案**：梯度为None、梯度累积、计算图太大等问题
13. **高级自动微分技术**：高阶导数、梯度检查

### 14.2 学习建议

1. **实践练习**：多进行自动微分的练习，熟悉各种场景下的梯度计算
2. **理解原理**：深入理解计算图、链式法则等概念
3. **性能优化**：了解如何优化自动微分的性能
4. **结合实际**：将自动微分与具体的深度学习任务结合起来

### 14.3 进阶学习

1. **分布式训练**：学习如何在多GPU上进行分布式训练
2. **混合精度训练**：学习如何使用混合精度训练来提高性能
3. **模型量化**：学习如何量化模型以提高推理速度
4. **自定义优化器**：学习如何自定义优化器

通过本章的学习，您应该已经掌握了PyTorch自动微分的基本用法，可以开始构建和训练深度学习模型了。