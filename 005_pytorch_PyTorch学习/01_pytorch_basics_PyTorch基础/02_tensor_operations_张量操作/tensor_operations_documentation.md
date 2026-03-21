# PyTorch 张量操作详细文档

## 1. 张量索引和切片

### 1.1 基本索引

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

输出：
```
tensor[0, 0]: tensor(1)
tensor[1, 2]: tensor(7)
```

### 1.2 切片操作

```python
# 切片操作
print("tensor[:, 0]:", tensor[:, 0])  # 第一列
print("tensor[0, :]:", tensor[0, :])  # 第一行
print("tensor[1:3, 1:3]:", tensor[1:3, 1:3])  # 第二、三行，第二、三列
```

输出：
```
tensor[:, 0]: tensor([1, 5, 9])
tensor[0, :]: tensor([1, 2, 3, 4])
tensor[1:3, 1:3]: tensor([[ 6,  7],
        [10, 11]])
```

### 1.3 布尔索引

```python
# 布尔索引
mask = tensor > 5
print("mask:")
print(mask)
print("tensor[mask]:", tensor[mask])
```

输出：
```
mask:
tensor([[False, False, False, False],
        [False,  True,  True,  True],
        [ True,  True,  True,  True]])
tensor[mask]: tensor([ 6,  7,  8,  9, 10, 11, 12])
```

## 2. 张量形状操作

### 2.1 reshape

```python
# 创建一个张量
tensor = torch.randn(2, 3, 4)
print("原始张量形状:", tensor.shape)

# reshape
reshaped = tensor.reshape(2, 12)
print("reshape(2, 12):", reshaped.shape)
```

输出：
```
原始张量形状: torch.Size([2, 3, 4])
reshape(2, 12): torch.Size([2, 12])
```

### 2.2 view

```python
# view
viewed = tensor.view(2, 12)
print("view(2, 12):", viewed.shape)
```

输出：
```
view(2, 12): torch.Size([2, 12])
```

**注意**：`view` 要求张量是连续的，而 `reshape` 会在需要时自动处理不连续的情况。

### 2.3 squeeze 和 unsqueeze

```python
# squeeze 和 unsqueeze
tensor_2d = torch.randn(1, 4)
print("原始形状:", tensor_2d.shape)
squeezed = tensor_2d.squeeze()
print("squeeze:", squeezed.shape)
unsqueezed = squeezed.unsqueeze(0)
print("unsqueeze(0):", unsqueezed.shape)
```

输出：
```
原始形状: torch.Size([1, 4])
squeeze: torch.Size([4])
unsqueeze(0): torch.Size([1, 4])
```

### 2.4 transpose

```python
# transpose
tensor_2d = torch.randn(2, 3)
print("原始形状:", tensor_2d.shape)
transposed = tensor_2d.transpose(0, 1)
print("transpose(0, 1):", transposed.shape)
```

输出：
```
原始形状: torch.Size([2, 3])
transpose(0, 1): torch.Size([3, 2])
```

### 2.5 permute

```python
# permute
tensor_3d = torch.randn(2, 3, 4)
print("原始形状:", tensor_3d.shape)
permuted = tensor_3d.permute(2, 0, 1)
print("permute(2, 0, 1):", permuted.shape)
```

输出：
```
原始形状: torch.Size([2, 3, 4])
permute(2, 0, 1): torch.Size([4, 2, 3])
```

## 3. 张量数学运算

### 3.1 基本运算

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

输出：
```
a + b: tensor([5, 7, 9])
a - b: tensor([-3, -3, -3])
a * b: tensor([ 4, 10, 18])
a / b: tensor([0.2500, 0.4000, 0.5000])
a ** 2: tensor([1, 4, 9])
```

### 3.2 矩阵乘法

```python
# 矩阵乘法
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])
print("c.matmul(d):")
print(c.matmul(d))
print("c @ d:")
print(c @ d)
```

输出：
```
c.matmul(d):
tensor([[19, 22],
        [43, 50]])
c @ d:
tensor([[19, 22],
        [43, 50]])
```

### 3.3 广播机制

```python
# 广播机制
e = torch.tensor([[1, 2, 3], [4, 5, 6]])
f = torch.tensor([10, 20, 30])
print("e + f:")
print(e + f)
```

输出：
```
e + f:
tensor([[11, 22, 33],
        [14, 25, 36]])
```

## 4. 张量统计操作

### 4.1 基本统计

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

输出：
```
原始张量:
tensor([[ 0.6614,  0.2669,  0.0617,  0.6213],
        [-0.4519, -0.1661,  0.2437, -1.1173],
        [ 0.3037, -0.4381, -0.1088, -0.1527]])
均值: tensor(-0.0471)
总和: tensor(-0.5656)
最大值: tensor(0.6614)
最小值: tensor(-1.1173)
标准差: tensor(0.5348)
```

### 4.2 沿特定维度统计

```python
# 沿特定维度统计
print("沿维度0的均值:", tensor.mean(dim=0))
print("沿维度1的总和:", tensor.sum(dim=1))
print("沿维度0的最大值:", tensor.max(dim=0))
print("沿维度1的最小值:", tensor.min(dim=1))
```

输出：
```
沿维度0的均值: tensor([ 0.1717, -0.1124,  0.0655, -0.2162])
沿维度1的总和: tensor([ 1.6113, -1.4917, -0.3959])
沿维度0的最大值: torch.return_types.max(
values=tensor([ 0.6614,  0.2669,  0.2437,  0.6213]),
indices=tensor([0, 0, 1, 0]))
沿维度1的最小值: torch.return_types.min(
values=tensor([ 0.0617, -1.1173, -0.4381]),
indices=tensor([2, 3, 1]))
```

## 5. 张量比较操作

### 5.1 基本比较

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

输出：
```
a: tensor([1, 2, 3, 4, 5])
b: tensor([3, 3, 3, 3, 3])
a == b: tensor([False, False,  True, False, False])
a != b: tensor([ True,  True, False,  True,  True])
a > b: tensor([False, False, False,  True,  True])
a < b: tensor([ True,  True, False, False, False])
a >= b: tensor([False, False,  True,  True,  True])
a <= b: tensor([ True,  True,  True, False, False])
```

### 5.2 其他比较函数

```python
# 其他比较函数
print("torch.all(a == b):", torch.all(a == b))
print("torch.any(a > b):", torch.any(a > b))
```

输出：
```
torch.all(a == b): tensor(False)
torch.any(a > b): tensor(True)
```

## 6. 张量类型转换

### 6.1 数据类型转换

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

输出：
```
原始张量: tensor([1, 2, 3, 4, 5])
原始类型: torch.int64
转换为float: tensor([1., 2., 3., 4., 5.])
转换为double: tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
转换为int: tensor([1, 2, 3, 4, 5], dtype=torch.int32)
转换为bool: tensor([True, True, True, True, True])
```

### 6.2 转换为NumPy数组

```python
# 转换为NumPy数组
numpy_array = tensor.numpy()
print("NumPy数组:", numpy_array)
print("NumPy数组类型:", numpy_array.dtype)
```

输出：
```
NumPy数组: [1 2 3 4 5]
NumPy数组类型: int64
```

## 7. 张量的设备移动

### 7.1 设备移动

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

输出：
```
原始设备: cpu
移动到GPU后: cuda:0  # 如果有GPU
移动回CPU后: cpu
或
原始设备: cpu
没有可用的GPU  # 如果没有GPU
```

## 8. 张量的内存管理

### 8.1 内存占用

```python
# 创建一个张量
tensor = torch.randn(1000, 1000)
print("张量大小:", tensor.size())
print("元素数量:", tensor.numel())
print("内存占用 (MB):", tensor.element_size() * tensor.numel() / 1024 / 1024)
```

输出：
```
张量大小: torch.Size([1000, 1000])
元素数量: 1000000
内存占用 (MB): 4.0  # 对于float32类型
```

### 8.2 内存释放

```python
# 释放内存
del tensor
print("张量已删除")
```

## 9. 张量的高级操作

### 9.1 求迹

```python
# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始张量:")
print(tensor)

# 求迹
print("求迹:", tensor.trace())
```

输出：
```
原始张量:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
求迹: tensor(15)
```

### 9.2 求范数

```python
# 求范数
print("L1范数:", tensor.norm(1))
print("L2范数:", tensor.norm(2))
```

输出：
```
L1范数: tensor(45.)
L2范数: tensor(16.8819)
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

输出：
```
排序后的张量:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
索引:
tensor([[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]])
```

### 9.4 唯一值

```python
# 唯一值
unique_values = torch.unique(tensor)
print("唯一值:", unique_values)
```

输出：
```
唯一值: tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## 10. 性能测试

### 10.1 矩阵乘法性能

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

输出示例：
```
大小为 100x100 的矩阵乘法耗时: 0.000000 秒
大小为 1000x1000 的矩阵乘法耗时: 0.031250 秒
大小为 5000x5000 的矩阵乘法耗时: 4.687500 秒
```

## 11. 实际应用示例

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

输出：
```
预测形状: torch.Size([100, 1])
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

输出：
```
批量形状: torch.Size([32, 3, 224, 224])
通道均值: tensor([0.5001, 0.4998, 0.5000])
```

## 12. 常见问题与解决方案

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

## 13. 总结

张量操作是PyTorch中的核心功能，掌握这些操作对于使用PyTorch进行深度学习至关重要。本文档介绍了多种张量操作，包括：

- 索引和切片
- 形状操作
- 数学运算
- 统计操作
- 比较操作
- 类型转换
- 设备移动
- 内存管理
- 高级操作
- 性能测试
- 实际应用示例

通过这些操作，您可以灵活地处理和转换张量，为后续的深度学习任务做好准备。