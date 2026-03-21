# PyTorch 张量创建教材

## 第一章：张量的概念

### 1.1 什么是张量

张量（Tensor）是PyTorch中的基本数据结构，是一个多维数组，类似于NumPy的ndarray，但具有以下特点：

- **GPU加速**：可以在GPU上运行，提高计算速度
- **自动微分**：支持自动计算梯度，是深度学习的核心
- **动态计算图**：允许在运行时修改计算图

### 1.2 张量的维度

| 维度 | 名称 | 示例 |
|------|------|------|
| 0 | 标量 (Scalar) | `torch.tensor(5)` |
| 1 | 向量 (Vector) | `torch.tensor([1, 2, 3])` |
| 2 | 矩阵 (Matrix) | `torch.tensor([[1, 2], [3, 4]])` |
| 3+ | 高维张量 | `torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` |

### 1.3 张量与NumPy数组的关系

- 张量和NumPy数组可以相互转换
- 从NumPy创建的张量与原数组共享内存
- 张量支持GPU，而NumPy只支持CPU

## 第二章：张量的创建方法

### 2.1 从Python列表创建

**一维张量**

```python
import torch

# 从Python列表创建一维张量
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
print(tensor)  # 输出: tensor([1, 2, 3, 4, 5])
```

**二维张量**

```python
# 从Python列表创建二维张量
data_2d = [[1, 2, 3], [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print(tensor_2d)  # 输出: tensor([[1, 2, 3], [4, 5, 6]])
```

**三维张量**

```python
# 从Python列表创建三维张量
data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
tensor_3d = torch.tensor(data_3d)
print(tensor_3d)
```

### 2.2 从NumPy数组创建

**基本用法**

```python
import numpy as np
import torch

# 从NumPy数组创建张量
np_array = np.array([1, 2, 3, 4, 5])
tensor_from_np = torch.from_numpy(np_array)
print(tensor_from_np)  # 输出: tensor([1, 2, 3, 4, 5])
```

**内存共享**

```python
# 演示内存共享
np_array = np.array([1, 2, 3, 4, 5])
tensor_from_np = torch.from_numpy(np_array)

# 修改NumPy数组
np_array[0] = 100
print(tensor_from_np)  # 输出: tensor([100,   2,   3,   4,   5])
```

### 2.3 创建特殊张量

**全零张量**

```python
# 创建全零张量
zeros = torch.zeros(2, 3)
print(zeros)  # 输出: tensor([[0., 0., 0.], [0., 0., 0.]])
```

**全一张量**

```python
# 创建全一张量
ones = torch.ones(3, 2)
print(ones)  # 输出: tensor([[1., 1.], [1., 1.], [1., 1.]])
```

**单位矩阵**

```python
# 创建单位矩阵
eye = torch.eye(4)
print(eye)  # 输出: 4x4单位矩阵
```

**指定值的张量**

```python
# 创建指定值的张量
full = torch.full((2, 2), 7)
print(full)  # 输出: tensor([[7, 7], [7, 7]])
```

**随机张量**

```python
# 创建随机张量（均匀分布）
rand = torch.rand(2, 3)
print(rand)  # 输出: 0-1之间的随机值

# 创建随机张量（正态分布）
randn = torch.randn(2, 3)
print(randn)  # 输出: 均值为0，标准差为1的随机值
```

**序列张量**

```python
# 创建整数序列张量
arange = torch.arange(0, 10, 2)
print(arange)  # 输出: tensor([0, 2, 4, 6, 8])

# 创建线性间隔张量
linspace = torch.linspace(0, 1, 5)
print(linspace)  # 输出: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

## 第三章：张量的属性

### 3.1 基本属性

```python
# 张量的属性
tensor = torch.rand(2, 3, 4)
print("形状:", tensor.shape)  # 输出: torch.Size([2, 3, 4])
print("维度:", tensor.ndim)    # 输出: 3
print("元素数量:", tensor.numel())  # 输出: 24
print("数据类型:", tensor.dtype)  # 输出: torch.float32
print("设备:", tensor.device)    # 输出: cpu
```

### 3.2 数据类型

| 数据类型 | 描述 | 别名 |
|---------|------|------|
| torch.float32 | 32位浮点数 | torch.float |
| torch.float64 | 64位浮点数 | torch.double |
| torch.int32 | 32位整数 | torch.int |
| torch.int64 | 64位整数 | torch.long |
| torch.bool | 布尔值 | - |

**指定数据类型**

```python
# 创建指定数据类型的张量
tensor_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(tensor_float32.dtype)  # 输出: torch.float32
```

**数据类型转换**

```python
# 数据类型转换
tensor_int = torch.tensor([1, 2, 3], dtype=torch.int64)
tensor_float = tensor_int.float()
print(tensor_float.dtype)  # 输出: torch.float32
```

### 3.3 设备

**检查设备**

```python
# 检查张量设备
tensor_cpu = torch.tensor([1, 2, 3])
print(tensor_cpu.device)  # 输出: cpu
```

**移动到GPU**

```python
# 检查是否有GPU
if torch.cuda.is_available():
    # 移动到GPU
    tensor_gpu = tensor_cpu.to('cuda')
    print(tensor_gpu.device)  # 输出: cuda:0
else:
    print("没有可用的GPU")
```

## 第四章：张量的可视化

### 4.1 二维张量可视化

```python
import matplotlib.pyplot as plt
import torch

# 创建一个二维张量
tensor_2d = torch.rand(10, 10)

# 可视化张量
plt.figure(figsize=(8, 6))
plt.imshow(tensor_2d.numpy(), cmap='viridis')
plt.colorbar()
plt.title('二维张量可视化')
plt.show()
```

### 4.2 一维张量可视化

```python
# 创建一个一维张量并可视化
tensor_1d = torch.linspace(0, 1, 100)
plt.figure(figsize=(8, 6))
plt.plot(tensor_1d.numpy())
plt.title('一维张量可视化')
plt.xlabel('索引')
plt.ylabel('值')
plt.show()
```

## 第五章：性能测试

### 5.1 张量创建时间测试

```python
import time
import torch

# 测试不同大小张量的创建时间
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    start = time.time()
    tensor = torch.rand(size)
    end = time.time()
    times.append(end - start)
    print(f"创建大小为 {size} 的张量耗时: {end - start:.6f} 秒")
```

### 5.2 性能优化建议

1. **使用适当的设备**：对于大型张量，使用GPU可以显著提高性能
2. **批处理**：将大型计算分解为小批量处理
3. **内存管理**：及时释放不需要的张量，避免内存泄漏
4. **数据类型选择**：根据精度要求选择合适的数据类型，如float32比float64内存占用少

## 第六章：实际应用示例

### 6.1 线性回归中的权重和偏置

```python
# 创建权重张量（需要梯度）
weights = torch.randn(3, 1, requires_grad=True)
# 创建偏置张量（需要梯度）
bias = torch.randn(1, requires_grad=True)
print("权重:", weights)
print("偏置:", bias)
```

### 6.2 图像数据

```python
# 创建一个随机的RGB图像 (高度, 宽度, 通道)
image = torch.rand(224, 224, 3)
print("图像形状:", image.shape)  # 输出: torch.Size([224, 224, 3])
```

### 6.3 批量数据

```python
# 创建一个批量的图像数据 (批量大小, 通道, 高度, 宽度)
batch = torch.rand(32, 3, 224, 224)
print("批量数据形状:", batch.shape)  # 输出: torch.Size([32, 3, 224, 224])
```

### 6.4 文本数据

```python
# 创建文本嵌入张量 (序列长度, 嵌入维度)
embedding = torch.rand(100, 512)
print("嵌入张量形状:", embedding.shape)  # 输出: torch.Size([100, 512])
```

## 第七章：常见问题与解决方案

### 7.1 内存错误

**问题**：创建大型张量时出现内存错误

**解决方案**：
- 使用GPU进行计算
- 分批处理数据
- 使用内存映射文件
- 减小张量大小或使用更高效的数据类型

### 7.2 数据类型不匹配

**问题**：不同数据类型的张量之间无法直接运算

**解决方案**：
- 在运算前统一数据类型
- 使用`.to()`方法转换数据类型
- 在创建张量时指定正确的数据类型

### 7.3 设备不匹配

**问题**：不同设备上的张量之间无法直接运算

**解决方案**：
- 在运算前将张量移动到同一设备
- 使用`.to()`方法移动张量
- 确保模型和数据在同一设备上

### 7.4 梯度计算问题

**问题**：张量没有梯度信息

**解决方案**：
- 在创建张量时设置`requires_grad=True`
- 使用`torch.autograd`跟踪梯度
- 确保计算图正确构建

## 第八章：习题

### 8.1 选择题

1. 以下哪个函数用于创建全零张量？
   A. torch.ones()
   B. torch.zeros()
   C. torch.full()
   D. torch.rand()

2. 从NumPy数组创建张量时，以下说法正确的是：
   A. 会创建一个新的内存副本
   B. 与原NumPy数组共享内存
   C. 只能创建一维张量
   D. 无法指定数据类型

3. 以下哪个方法可以将张量移动到GPU？
   A. tensor.cuda()
   B. tensor.to('cuda')
   C. torch.cuda(tensor)
   D. 以上都可以

### 8.2 填空题

1. 创建一个形状为(2, 3)的全一张量，应该使用函数 __________。
2. 张量的形状可以通过 __________ 属性获取。
3. 要在创建张量时指定数据类型，应该使用 __________ 参数。

### 8.3 简答题

1. 张量和NumPy数组的区别是什么？
2. 如何检查张量所在的设备？
3. 什么是内存共享？在什么情况下会发生？

### 8.4 编程题

1. 创建一个形状为(3, 4)的随机张量，并将其转换为NumPy数组。
2. 创建一个形状为(2, 2)的单位矩阵，并计算其行列式。
3. 创建一个从0到10，步长为0.5的一维张量。
4. 创建一个形状为(5, 5)的二维张量，其中每个元素都是7。

## 第九章：总结

### 9.1 知识回顾

1. **张量的概念**：张量是PyTorch中的基本数据结构，支持GPU加速和自动微分
2. **张量的创建方法**：从Python列表、NumPy数组创建，或创建特殊张量
3. **张量的属性**：形状、维度、数据类型、设备等
4. **张量的可视化**：使用Matplotlib可视化张量
5. **性能测试**：测试不同大小张量的创建时间
6. **实际应用**：线性回归、图像数据、批量数据等
7. **常见问题**：内存错误、数据类型不匹配、设备不匹配等

### 9.2 学习建议

1. **实践练习**：多创建不同类型的张量，熟悉各种创建方法
2. **理解内存管理**：注意内存共享和设备移动的影响
3. **性能优化**：了解如何优化张量操作的性能
4. **结合实际应用**：将张量创建与具体的深度学习任务结合起来

### 9.3 进阶学习

1. **张量操作**：学习张量的索引、切片、形状操作等
2. **自动微分**：深入了解PyTorch的自动微分机制
3. **神经网络**：使用张量构建和训练神经网络
4. **模型保存和加载**：学习如何保存和加载张量和模型

通过本章的学习，您应该已经掌握了PyTorch张量的基本创建方法和相关概念，为后续的深度学习学习打下了基础。