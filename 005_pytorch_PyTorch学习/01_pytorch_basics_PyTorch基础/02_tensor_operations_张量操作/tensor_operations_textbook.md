# PyTorch 张量操作教材

## 第一章：张量索引和切片

### 1.1 基本索引

张量的索引和切片操作与Python列表类似，可以通过索引访问张量中的元素。

```python
import torch

# 创建一个3x4的张量
tensor = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])

# 索引单个元素
print("tensor[0, 0]:", tensor[0, 0])  # 第一行第一列
print("tensor[1, 2]:", tensor[1, 2])  # 第二行第三列
```

### 1.2 切片操作

切片操作可以访问张量的一个子集。

```python
# 切片操作
print("tensor[:, 0]:", tensor[:, 0])  # 第一列
print("tensor[0, :]:", tensor[0, :])  # 第一行
print("tensor[1:3, 1:3]:", tensor[1:3, 1:3])  # 第二、三行，第二、三列
```

### 1.3 布尔索引

布尔索引可以根据条件选择张量中的元素。

```python
# 布尔索引
mask = tensor > 5
print("mask:")
print(mask)
print("tensor[mask]:", tensor[mask])
```

## 第二章：张量形状操作

### 2.1 reshape

`reshape` 方法可以改变张量的形状，但不改变张量的数据。

```python
# 创建一个张量
tensor = torch.randn(2, 3, 4)
print("原始张量形状:", tensor.shape)

# reshape
reshaped = tensor.reshape(2, 12)
print("reshape(2, 12):", reshaped.shape)
```

### 2.2 view

`view` 方法与 `reshape` 类似，但要求张量是连续的。

```python
# view
viewed = tensor.view(2, 12)
print("view(2, 12):", viewed.shape)
```

### 2.3 squeeze 和 unsqueeze

- `squeeze`：移除大小为1的维度
- `unsqueeze`：添加大小为1的维度

```python
# squeeze 和 unsqueeze
tensor_2d = torch.randn(1, 4)
print("原始形状:", tensor_2d.shape)
squeezed = tensor_2d.squeeze()
print("squeeze:", squeezed.shape)
unsqueezed = squeezed.unsqueeze(0)
print("unsqueeze(0):", unsqueezed.shape)
```

### 2.4 transpose

`transpose` 方法可以交换张量的两个维度。

```python
# transpose
tensor_2d = torch.randn(2, 3)
print("原始形状:", tensor_2d.shape)
transposed = tensor_2d.transpose(0, 1)
print("transpose(0, 1):", transposed.shape)
```

### 2.5 permute

`permute` 方法可以重新排列张量的所有维度。

```python
# permute
tensor_3d = torch.randn(2, 3, 4)
print("原始形状:", tensor_3d.shape)
permuted = tensor_3d.permute(2, 0, 1)
print("permute(2, 0, 1):", permuted.shape)
```

## 第三章：张量数学运算

### 3.1 基本运算

PyTorch支持常见的数学运算，如加法、减法、乘法、除法等。

```python
# 创建两个张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 基本运算
print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print("a ** 2:", a ** 2)
```

### 3.2 矩阵乘法

矩阵乘法是深度学习中最常用的运算之一。

```python
# 矩阵乘法
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])
print("c.matmul(d):")
print(c.matmul(d))
print("c @ d:")
print(c @ d)
```

### 3.3 广播机制

广播机制允许不同形状的张量进行运算。

```python
# 广播机制
e = torch.tensor([[1, 2, 3], [4, 5, 6]])
f = torch.tensor([10, 20, 30])
print("e + f:")
print(e + f)
```

## 第四章：张量统计操作

### 4.1 基本统计

PyTorch提供了多种统计函数，如均值、总和、最大值、最小值、标准差等。

```python
# 创建一个张量
tensor = torch.randn(3, 4)
print("原始张量:")
print(tensor)

# 基本统计
print("均值:", tensor.mean())
print("总和:", tensor.sum())
print("最大值:", tensor.max())
print("最小值:", tensor.min())
print("标准差:", tensor.std())
```

### 4.2 沿特定维度统计

可以沿特定维度进行统计操作。

```python
# 沿特定维度统计
print("沿维度0的均值:", tensor.mean(dim=0))
print("沿维度1的总和:", tensor.sum(dim=1))
print("沿维度0的最大值:", tensor.max(dim=0))
print("沿维度1的最小值:", tensor.min(dim=1))
```

## 第五章：张量比较操作

### 5.1 基本比较

PyTorch支持常见的比较操作，如等于、不等于、大于、小于等。

```python
# 创建两个张量
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([3, 3, 3, 3, 3])

print("a:", a)
print("b:", b)

# 比较操作
print("a == b:", a == b)
print("a != b:", a != b)
print("a > b:", a > b)
print("a < b:", a < b)
print("a >= b:", a >= b)
print("a <= b:", a <= b)
```

### 5.2 其他比较函数

```python
# 其他比较函数
print("torch.all(a == b):", torch.all(a == b))
print("torch.any(a > b):", torch.any(a > b))
```

## 第六章：张量类型转换

### 6.1 数据类型转换

PyTorch支持多种数据类型，可以在不同类型之间转换。

```python
# 创建一个张量
tensor = torch.tensor([1, 2, 3, 4, 5])
print("原始张量:", tensor)
print("原始类型:", tensor.dtype)

# 类型转换
print("转换为float:", tensor.float())
print("转换为double:", tensor.double())
print("转换为int:", tensor.int())
print("转换为bool:", tensor.bool())
```

### 6.2 转换为NumPy数组

```python
# 转换为NumPy数组
numpy_array = tensor.numpy()
print("NumPy数组:", numpy_array)
print("NumPy数组类型:", numpy_array.dtype)
```

## 第七章：张量的设备移动

### 7.1 设备移动

PyTorch可以在CPU和GPU之间移动张量。

```python
# 创建一个CPU张量
tensor = torch.tensor([1, 2, 3, 4, 5])
print("原始设备:", tensor.device)

# 检查是否有GPU
if torch.cuda.is_available():
    # 移动到GPU
    tensor_gpu = tensor.to('cuda')
    print("移动到GPU后:", tensor_gpu.device)
    
    # 移动回CPU
    tensor_cpu = tensor_gpu.to('cpu')
    print("移动回CPU后:", tensor_cpu.device)
else:
    print("没有可用的GPU")
```

## 第八章：张量的内存管理

### 8.1 内存占用

了解张量的内存占用对于优化模型性能很重要。

```python
# 创建一个张量
tensor = torch.randn(1000, 1000)
print("张量大小:", tensor.size())
print("元素数量:", tensor.numel())
print("内存占用 (MB):", tensor.element_size() * tensor.numel() / 1024 / 1024)
```

### 8.2 内存释放

及时释放不需要的张量可以节省内存。

```python
# 释放内存
del tensor
print("张量已删除")
```

## 第九章：张量的高级操作

### 9.1 求迹

矩阵的迹是主对角线上元素的和。

```python
# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始张量:")
print(tensor)

# 求迹
print("求迹:", tensor.trace())
```

### 9.2 求范数

```python
# 求范数
print("L1范数:", tensor.norm(1))
print("L2范数:", tensor.norm(2))
```

### 9.3 排序

```python
# 排序
sorted_tensor, indices = torch.sort(tensor, dim=1)
print("排序后的张量:")
print(sorted_tensor)
print("索引:")
print(indices)
```

### 9.4 唯一值

```python
# 唯一值
unique_values = torch.unique(tensor)
print("唯一值:", unique_values)
```

## 第十章：性能测试

### 10.1 矩阵乘法性能

测试不同大小矩阵的乘法性能。

```python
import time

# 测试不同大小张量的运算性能
sizes = [100, 1000, 5000]
times = []

for size in sizes:
    # 创建两个随机张量
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    
    # 测试矩阵乘法性能
    start = time.time()
    c = a @ b
    end = time.time()
    
    times.append(end - start)
    print(f"大小为 {size}x{size} 的矩阵乘法耗时: {end - start:.6f} 秒")
```

## 第十一章：实际应用示例

### 11.1 线性回归

```python
# 示例1：线性回归
# 创建输入和目标
X = torch.randn(100, 3)
y = torch.randn(100, 1)
# 创建权重和偏置
w = torch.randn(3, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
# 前向传播
y_pred = X @ w + b
print("预测形状:", y_pred.shape)
```

### 11.2 图像处理

```python
# 示例2：图像处理
# 创建一个批量的图像数据 (批量大小, 通道, 高度, 宽度)
batch = torch.rand(32, 3, 224, 224)
print("批量形状:", batch.shape)
# 计算每个通道的均值
channel_means = batch.mean(dim=(0, 2, 3))
print("通道均值:", channel_means)
```

## 第十二章：常见问题与解决方案

### 12.1 形状不匹配

**问题**：进行张量运算时出现形状不匹配错误

**解决方案**：
- 检查张量的形状是否兼容
- 使用 `reshape` 或 `view` 调整张量形状
- 利用广播机制处理不同形状的张量

### 12.2 内存不足

**问题**：创建大型张量时出现内存不足错误

**解决方案**：
- 使用GPU进行计算
- 分批处理数据
- 使用更高效的数据类型（如float16）
- 及时释放不需要的张量

### 12.3 设备不匹配

**问题**：不同设备上的张量之间无法直接运算

**解决方案**：
- 在运算前将张量移动到同一设备
- 使用 `.to()` 方法移动张量

### 12.4 类型不匹配

**问题**：不同数据类型的张量之间无法直接运算

**解决方案**：
- 在运算前统一数据类型
- 使用 `.to()` 方法转换数据类型

## 第十三章：习题

### 13.1 选择题

1. 以下哪个方法可以交换张量的两个维度？
   A. reshape
   B. view
   C. transpose
   D. permute

2. 以下哪个方法可以移除大小为1的维度？
   A. squeeze
   B. unsqueeze
   C. flatten
   D. ravel

3. 以下哪个操作会返回排序后的张量和索引？
   A. torch.sort()
   B. torch.argsort()
   C. torch.max()
   D. torch.min()

### 13.2 填空题

1. 要计算张量的均值，应该使用 __________ 方法。
2. 要将张量移动到GPU，应该使用 __________ 方法。
3. 要获取张量的元素数量，应该使用 __________ 属性。

### 13.3 简答题

1. 解释广播机制的工作原理。
2. 比较 `reshape` 和 `view` 的区别。
3. 如何计算张量沿特定维度的最大值？

### 13.4 编程题

1. 创建一个形状为(3, 4)的张量，然后：
   - 提取第一行和第三行
   - 提取所有行的第二列
   - 提取中心2x2的子张量

2. 创建两个形状分别为(2, 3)和(3, 4)的张量，计算它们的矩阵乘法。

3. 创建一个形状为(2, 3, 4)的张量，然后：
   - 将其reshape为(2, 12)
   - 交换第0维和第1维
   - 计算沿第2维的均值

4. 创建一个形状为(100, 100)的随机张量，计算：
   - 整个张量的均值和标准差
   - 沿第0维的最大值和最小值
   - 矩阵的迹和L2范数

## 第十四章：总结

### 14.1 知识回顾

1. **张量索引和切片**：基本索引、切片操作、布尔索引
2. **张量形状操作**：reshape、view、squeeze、unsqueeze、transpose、permute
3. **张量数学运算**：基本运算、矩阵乘法、广播机制
4. **张量统计操作**：均值、总和、最大值、最小值、标准差
5. **张量比较操作**：基本比较、其他比较函数
6. **张量类型转换**：数据类型转换、转换为NumPy数组
7. **张量的设备移动**：CPU和GPU之间的移动
8. **张量的内存管理**：内存占用、内存释放
9. **张量的高级操作**：求迹、求范数、排序、唯一值
10. **性能测试**：矩阵乘法性能测试
11. **实际应用**：线性回归、图像处理

### 14.2 学习建议

1. **实践练习**：多进行张量操作的练习，熟悉各种操作的用法
2. **理解原理**：深入理解广播机制、内存管理等概念
3. **性能优化**：了解如何优化张量操作的性能
4. **结合实际**：将张量操作与具体的深度学习任务结合起来

### 14.3 进阶学习

1. **自动微分**：学习PyTorch的自动微分机制
2. **神经网络**：使用张量构建和训练神经网络
3. **模型优化**：学习如何优化模型的性能和内存使用
4. **分布式训练**：学习如何在多GPU上进行分布式训练

通过本章的学习，您应该已经掌握了PyTorch张量的各种操作，为后续的深度学习学习打下了基础。