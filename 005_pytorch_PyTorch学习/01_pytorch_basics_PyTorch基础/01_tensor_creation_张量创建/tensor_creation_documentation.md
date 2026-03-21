# PyTorch 张量创建详细文档

## 1. 张量的基本概念

张量（Tensor）是PyTorch中的基本数据结构，类似于NumPy的ndarray，但具有以下特点：
- 支持GPU加速
- 支持自动微分
- 是构建神经网络的基本单位

## 2. 从Python列表创建张量

### 2.1 一维张量

```python
import torch

# 从Python列表创建一维张量
data = [1, 2, 3, 4, 5]
tensor = torch.tensor(data)
print("一维张量:", tensor)
print("张量形状:", tensor.shape)
```

输出：
```
一维张量: tensor([1, 2, 3, 4, 5])
张量形状: torch.Size([5])
```

### 2.2 二维张量

```python
# 从Python列表创建二维张量
data_2d = [[1, 2, 3], [4, 5, 6]]
tensor_2d = torch.tensor(data_2d)
print("二维张量:", tensor_2d)
print("张量形状:", tensor_2d.shape)
```

输出：
```
二维张量: tensor([[1, 2, 3],
        [4, 5, 6]])
张量形状: torch.Size([2, 3])
```

### 2.3 三维张量

```python
# 从Python列表创建三维张量
data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
tensor_3d = torch.tensor(data_3d)
print("三维张量:", tensor_3d)
print("张量形状:", tensor_3d.shape)
```

输出：
```
三维张量: tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])
张量形状: torch.Size([2, 2, 2])
```

## 3. 从NumPy数组创建张量

### 3.1 基本用法

```python
import numpy as np
import torch

# 从NumPy数组创建张量
np_array = np.array([1, 2, 3, 4, 5])
tensor_from_np = torch.from_numpy(np_array)
print("从NumPy创建的张量:", tensor_from_np)
```

输出：
```
从NumPy创建的张量: tensor([1, 2, 3, 4, 5])
```

### 3.2 内存共享

**重要**：从NumPy创建的张量与原NumPy数组共享内存，修改其中一个会影响另一个。

```python
# 演示内存共享
np_array = np.array([1, 2, 3, 4, 5])
tensor_from_np = torch.from_numpy(np_array)

# 修改NumPy数组
np_array[0] = 100
print("修改NumPy数组后，张量也会变化:", tensor_from_np)
```

输出：
```
修改NumPy数组后，张量也会变化: tensor([100,   2,   3,   4,   5])
```

## 4. 创建特殊张量

### 4.1 全零张量

```python
# 创建全零张量
zeros = torch.zeros(2, 3)
print("全零张量:", zeros)
```

输出：
```
全零张量: tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### 4.2 全一张量

```python
# 创建全一张量
ones = torch.ones(3, 2)
print("全一张量:", ones)
```

输出：
```
全一张量: tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
```

### 4.3 单位矩阵

```python
# 创建单位矩阵
eye = torch.eye(4)
print("单位矩阵:", eye)
```

输出：
```
单位矩阵: tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
```

### 4.4 指定值的张量

```python
# 创建指定值的张量
full = torch.full((2, 2), 7)
print("指定值的张量:", full)
```

输出：
```
指定值的张量: tensor([[7, 7],
        [7, 7]])
```

### 4.5 随机张量

```python
# 创建随机张量（均匀分布）
rand = torch.rand(2, 3)
print("随机张量（均匀分布）:", rand)

# 创建随机张量（正态分布）
randn = torch.randn(2, 3)
print("随机张量（正态分布）:", randn)
```

输出：
```
随机张量（均匀分布）: tensor([[0.4963, 0.7682, 0.0885],
        [0.1320, 0.3074, 0.6341]])
随机张量（正态分布）: tensor([[ 0.3002, -0.1445, -1.1962],
        [ 0.1389, -1.4634,  0.6282]])
```

### 4.6 序列张量

```python
# 创建整数序列张量
arange = torch.arange(0, 10, 2)
print("整数序列张量:", arange)

# 创建线性间隔张量
linspace = torch.linspace(0, 1, 5)
print("线性间隔张量:", linspace)
```

输出：
```
整数序列张量: tensor([0, 2, 4, 6, 8])
线性间隔张量: tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

## 5. 张量的数据类型

### 5.1 常用数据类型

| 数据类型 | 描述 | 别名 |
|---------|------|------|
| torch.float32 | 32位浮点数 | torch.float |
| torch.float64 | 64位浮点数 | torch.double |
| torch.int32 | 32位整数 | torch.int |
| torch.int64 | 64位整数 | torch.long |
| torch.bool | 布尔值 | - |

### 5.2 指定数据类型

```python
# 创建指定数据类型的张量
tensor_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
print("float32张量:", tensor_float32, "数据类型:", tensor_float32.dtype)

tensor_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
print("int64张量:", tensor_int64, "数据类型:", tensor_int64.dtype)
```

输出：
```
float32张量: tensor([1., 2., 3.]) 数据类型: torch.float32
int64张量: tensor([1, 2, 3]) 数据类型: torch.int64
```

### 5.3 数据类型转换

```python
# 数据类型转换
tensor_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
tensor_converted = tensor_int64.float()
print("转换为float的张量:", tensor_converted, "数据类型:", tensor_converted.dtype)
```

输出：
```
转换为float的张量: tensor([1., 2., 3.]) 数据类型: torch.float32
```

## 6. 张量的设备

### 6.1 设备类型

- **CPU**：默认设备
- **CUDA**：GPU设备

### 6.2 检查设备

```python
# 检查张量设备
tensor_cpu = torch.tensor([1, 2, 3])
print("CPU张量设备:", tensor_cpu.device)
```

输出：
```
CPU张量设备: cpu
```

### 6.3 移动张量到GPU

```python
# 检查是否有GPU
if torch.cuda.is_available():
    # 移动到GPU
    tensor_gpu = tensor_cpu.to('cuda')
    print("GPU张量设备:", tensor_gpu.device)
else:
    print("没有可用的GPU")
```

输出：
```
GPU张量设备: cuda:0  # 如果有GPU
或
没有可用的GPU  # 如果没有GPU
```

## 7. 张量的属性

### 7.1 基本属性

```python
# 张量的属性
tensor = torch.rand(2, 3, 4)
print("张量:", tensor)
print("形状:", tensor.shape)
print("维度:", tensor.ndim)
print("元素数量:", tensor.numel())
print("数据类型:", tensor.dtype)
print("设备:", tensor.device)
```

输出：
```
张量: tensor([[[0.7066, 0.5448, 0.1332, 0.1387],
         [0.2404, 0.9132, 0.7489, 0.6117],
         [0.6545, 0.2316, 0.8637, 0.4556]],

        [[0.3312, 0.7181, 0.5685, 0.6531],
         [0.8230, 0.2072, 0.4043, 0.5575],
         [0.0205, 0.9040, 0.0293, 0.3890]]])
形状: torch.Size([2, 3, 4])
维度: 3
元素数量: 24
数据类型: torch.float32
设备: cpu
```

## 8. 张量的可视化

### 8.1 二维张量可视化

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

### 8.2 一维张量可视化

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

## 9. 性能测试

### 9.1 张量创建时间测试

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

输出示例：
```
创建大小为 1000 的张量耗时: 0.000000 秒
创建大小为 10000 的张量耗时: 0.000000 秒
创建大小为 100000 的张量耗时: 0.001000 秒
创建大小为 1000000 的张量耗时: 0.004000 秒
```

## 10. 实际应用示例

### 10.1 线性回归中的权重和偏置

```python
# 创建权重张量
weights = torch.randn(3, 1, requires_grad=True)
# 创建偏置张量
bias = torch.randn(1, requires_grad=True)
print("权重:", weights)
print("偏置:", bias)
```

输出：
```
权重: tensor([[0.1234],
        [0.5678],
        [0.9012]], requires_grad=True)
偏置: tensor([0.3456], requires_grad=True)
```

### 10.2 图像数据

```python
# 创建一个随机的RGB图像 (高度, 宽度, 通道)
image = torch.rand(224, 224, 3)
print("图像形状:", image.shape)
print("图像数据类型:", image.dtype)
```

输出：
```
图像形状: torch.Size([224, 224, 3])
图像数据类型: torch.float32
```

### 10.3 批量数据

```python
# 创建一个批量的图像数据 (批量大小, 通道, 高度, 宽度)
batch = torch.rand(32, 3, 224, 224)
print("批量数据形状:", batch.shape)
```

输出：
```
批量数据形状: torch.Size([32, 3, 224, 224])
```

## 11. 常见问题与解决方案

### 11.1 内存错误

**问题**：创建大型张量时出现内存错误

**解决方案**：
- 使用适当的设备（GPU有更大的内存）
- 分批处理数据
- 使用内存映射文件

### 11.2 数据类型不匹配

**问题**：不同数据类型的张量之间无法直接运算

**解决方案**：
- 在运算前统一数据类型
- 使用`.to()`方法转换数据类型

### 11.3 设备不匹配

**问题**：不同设备上的张量之间无法直接运算

**解决方案**：
- 在运算前将张量移动到同一设备
- 使用`.to()`方法移动张量

## 12. 总结

张量是PyTorch的核心数据结构，掌握张量的创建方法对于使用PyTorch进行深度学习至关重要。本文档介绍了多种创建张量的方法，包括：

- 从Python列表创建
- 从NumPy数组创建
- 创建特殊张量（全零、全一、随机等）
- 张量的数据类型和设备
- 张量的属性和可视化
- 性能测试和实际应用示例

通过这些方法，您可以根据不同的需求创建合适的张量，为后续的深度学习任务做好准备。