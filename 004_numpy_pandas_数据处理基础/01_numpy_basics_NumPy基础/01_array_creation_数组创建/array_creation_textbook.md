# 数组创建教材

## 第一章：数组创建基础

### 1.1 什么是数组

数组是NumPy中的核心数据结构，是一种有序的元素集合，所有元素具有相同的数据类型。

### 1.2 数组的特点

- **同质性**：所有元素类型相同
- **固定大小**：创建后大小不可变
- **多维支持**：支持1D、2D、3D等多维数组
- **高性能**：基于C语言实现，运算速度快

## 第二章：基本创建方法

### 2.1 从Python序列创建

```python
import numpy as np

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print("一维数组:", arr1)
print("形状:", arr1.shape)
print("维度:", arr1.ndim)

# 从元组创建
arr2 = np.array((6, 7, 8, 9, 10))
print("从元组创建的数组:", arr2)

# 创建二维数组
two_d_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("二维数组:")
print(two_d_array)
print("形状:", two_d_array.shape)
print("维度:", two_d_array.ndim)
```

### 2.2 特殊数组创建

```python
# 全零数组
zeros = np.zeros((2, 3))
print("全零数组:")
print(zeros)

# 全一数组
ones = np.ones((3, 2))
print("\n全一数组:")
print(ones)

# 单位矩阵
identity = np.eye(4)
print("\n单位矩阵:")
print(identity)

# 等差数列
arange = np.arange(0, 10, 2)
print("\n等差数列 (0-10, 步长2):", arange)

# 等间距数组
linspace = np.linspace(0, 1, 5)
print("\n等间距数组 (0-1, 5个元素):", linspace)
```

## 第三章：数据类型

### 3.1 常用数据类型

- **整数类型**：int8, int16, int32, int64
- **浮点类型**：float16, float32, float64
- **复数类型**：complex64, complex128
- **布尔类型**：bool
- **字符串类型**：string_

### 3.2 数据类型指定

```python
# 整数数组
int_array = np.array([1, 2, 3], dtype=np.int32)
print("整数数组:", int_array, "数据类型:", int_array.dtype)

# 浮点数数组
float_array = np.array([1, 2, 3], dtype=np.float64)
print("浮点数数组:", float_array, "数据类型:", float_array.dtype)

# 复数数组
complex_array = np.array([1, 2, 3], dtype=np.complex128)
print("复数数组:", complex_array, "数据类型:", complex_array.dtype)
```

## 第四章：从现有数组创建

### 4.1 复制操作

```python
original = np.array([1, 2, 3, 4, 5])

# 浅复制
shallow_copy = original.view()
print("浅复制:", shallow_copy)

# 深复制
deep_copy = original.copy()
print("深复制:", deep_copy)
```

### 4.2 数组变形

```python
# 重塑数组
reshape_array = original.reshape(5, 1)
print("重塑为5x1数组:")
print(reshape_array)

# 展平数组
flattened = original.flatten()
print("展平数组:", flattened)
```

## 第五章：随机数组生成

### 5.1 均匀分布

```python
# 0-1之间的均匀分布
random_uniform = np.random.rand(2, 3)
print("均匀分布随机数 (2x3):")
print(random_uniform)
```

### 5.2 正态分布

```python
# 标准正态分布
random_normal = np.random.randn(2, 3)
print("正态分布随机数 (2x3):")
print(random_normal)
```

### 5.3 整数随机数

```python
# 0-9之间的随机整数
random_int = np.random.randint(0, 10, size=(2, 3))
print("0-9之间的随机整数 (2x3):")
print(random_int)
```

## 第六章：性能优化

### 6.1 内存管理

```python
# 预分配内存
large_array = np.zeros((1000, 1000))
print("预分配内存的大型数组形状:", large_array.shape)

# 选择合适的数据类型
small_int_array = np.array([1, 2, 3], dtype=np.int8)
print("使用int8的内存占用:", small_int_array.nbytes, "字节")

normal_int_array = np.array([1, 2, 3], dtype=np.int32)
print("使用int32的内存占用:", normal_int_array.nbytes, "字节")
```

### 6.2 速度优化

```python
import time

# 测试向量化操作速度
start = time.time()
vectorized = np.arange(1000000) * 2
end = time.time()
print("向量化操作时间:", end - start, "秒")

# 测试循环操作速度
start = time.time()
loop_result = []
for i in range(1000000):
    loop_result.append(i * 2)
end = time.time()
print("循环操作时间:", end - start, "秒")
```

## 第七章：应用场景

### 7.1 数据初始化

- **神经网络权重初始化**：使用随机数组初始化网络权重
- **矩阵运算**：创建适当形状的矩阵进行线性代数运算
- **统计分析**：准备数据进行统计计算

### 7.2 测试数据生成

- **算法测试**：生成各种类型的测试数据
- **性能测试**：生成大型数据集测试算法性能
- **可视化**：生成用于绘图的数据

### 7.3 数值模拟

- **随机过程**：模拟随机事件
- **蒙特卡洛方法**：进行概率模拟
- **数值积分**：通过采样进行数值计算

## 第八章：最佳实践

### 8.1 数组创建最佳实践

1. **根据需求选择合适的创建方法**
   - 全零/全一数组：使用zeros()/ones()
   - 等差数列：使用arange()
   - 等间距数组：使用linspace()
   - 单位矩阵：使用eye()

2. **显式指定数据类型**
   - 节省内存：选择合适的精度
   - 避免类型转换：提前指定类型

3. **预分配内存**
   - 大型数组：先创建空数组再填充
   - 循环操作：避免动态扩展

4. **使用向量化操作**
   - 避免Python循环
   - 利用NumPy的C语言实现

### 8.2 常见问题及解决方案

1. **内存不足**
   - 解决方案：使用更高效的数据类型，分批处理数据

2. **类型错误**
   - 解决方案：显式指定数据类型，检查输入数据

3. **形状错误**
   - 解决方案：使用reshape()调整形状，检查输入维度

4. **性能问题**
   - 解决方案：使用向量化操作，预分配内存

## 第九章：习题

### 9.1 选择题

1. 以下哪个函数用于创建全零数组？
   - A) np.ones()
   - B) np.zeros()
   - C) np.empty()
   - D) np.eye()

2. 以下哪个函数用于创建等差数列？
   - A) np.linspace()
   - B) np.arange()
   - C) np.random.rand()
   - D) np.eye()

3. 以下哪个数据类型内存占用最小？
   - A) int8
   - B) int32
   - C) float32
   - D) float64

### 9.2 填空题

1. NumPy的核心数据结构是________________。
2. 创建单位矩阵使用的函数是________________。
3. 从现有数组创建新数组的深复制方法是________________。

### 9.3 简答题

1. 简述NumPy数组与Python列表的区别。
2. 简述不同数据类型的内存占用情况。
3. 简述如何优化数组创建的性能。

### 9.4 编程题

1. 创建一个3x3的单位矩阵。
2. 创建一个从0到100，步长为5的等差数列。
3. 创建一个2x4的随机数组，元素范围在0到1之间。
4. 创建一个包含10个元素的等间距数组，范围从-1到1。

## 第十章：总结

### 10.1 知识回顾

1. **数组创建方法**：从序列、特殊函数、随机生成
2. **数据类型**：整数、浮点数、复数、布尔值
3. **内存管理**：预分配内存、数据类型选择
4. **性能优化**：向量化操作、避免循环
5. **应用场景**：数据初始化、测试数据、数值模拟

### 10.2 学习建议

1. **实践练习**：尝试创建不同类型的数组
2. **性能测试**：比较不同创建方法的性能
3. **内存分析**：了解不同数据类型的内存占用
4. **应用开发**：在实际项目中应用数组创建技巧

### 10.3 进阶学习

1. **数组索引和切片**
2. **数组运算**
3. **广播机制**
4. **线性代数运算**