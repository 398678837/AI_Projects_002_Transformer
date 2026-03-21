# 数组操作教材

## 第一章：数组操作基础

### 1.1 什么是数组操作

数组操作是指对NumPy数组进行的各种操作，包括访问、修改、组合、分割等。这些操作是NumPy强大功能的基础，也是高效数据处理的关键。

### 1.2 数组操作的重要性

- **高效访问**：快速访问和修改数组元素
- **灵活处理**：灵活调整数组结构
- **数据组合**：方便地组合多个数据集
- **性能优化**：利用NumPy的C语言实现提高性能

## 第二章：索引和切片

### 2.1 基本索引

```python
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 访问单个元素
print("arr[0, 0] =", arr[0, 0])  # 1
print("arr[1, 2] =", arr[1, 2])  # 7

# 访问整行
print("arr[1, :] =", arr[1, :])  # [5, 6, 7, 8]

# 访问整列
print("arr[:, 1] =", arr[:, 1])  # [2, 6, 10]
```

### 2.2 切片操作

```python
# 切片操作
print("arr[0:2, 1:3] =")
print(arr[0:2, 1:3])  # 前两行，中间两列
# 输出:
# [[2, 3]
#  [6, 7]]

# 步长切片
print("arr[::2, ::2] =")
print(arr[::2, ::2])  # 每隔一行一列
# 输出:
# [[1, 3]
#  [9, 11]]
```

### 2.3 布尔索引

```python
# 布尔索引
mask = arr > 5
print("掩码:")
print(mask)
print("arr[mask] =", arr[mask])  # 大于5的元素

# 复合条件
mask = (arr > 5) & (arr < 10)
print("复合条件掩码:")
print(mask)
print("arr[mask] =", arr[mask])  # 大于5且小于10的元素
```

### 2.4 花式索引

```python
# 花式索引
rows = [0, 1, 2]
cols = [1, 2, 3]
print("arr[rows, cols] =", arr[rows, cols])  # [2, 7, 12]

# 列索引
print("arr[:, [0, 2]] =")
print(arr[:, [0, 2]])  # 第一列和第三列
```

## 第三章：形状操作

### 3.1 重塑数组

```python
arr = np.arange(12)
print("原始数组:", arr)
print("形状:", arr.shape)

# 重塑为3x4数组
reshaped = arr.reshape(3, 4)
print("\n重塑为3x4:")
print(reshaped)
print("形状:", reshaped.shape)

# 自动计算维度
reshaped = arr.reshape(3, -1)  # 自动计算第二维度
print("\n自动计算维度:")
print(reshaped.shape)  # (3, 4)
```

### 3.2 展平数组

```python
# 展平数组（返回副本）
flattened = reshaped.flatten()
print("\n展平 (flatten):", flattened)
print("形状:", flattened.shape)

# 展平数组（返回视图）
raveled = reshaped.ravel()
print("\n展平 (ravel):", raveled)
print("形状:", raveled.shape)
```

### 3.3 转置

```python
# 转置
print("\n转置:")
print(reshaped.T)
print("形状:", reshaped.T.shape)

# 使用transpose
print("\n使用transpose:")
print(reshaped.transpose())
```

## 第四章：数组拼接

### 4.1 垂直拼接

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("数组a:")
print(a)
print("数组b:")
print(b)

# 垂直拼接
vstack = np.vstack((a, b))
print("\n垂直拼接:")
print(vstack)
print("形状:", vstack.shape)
```

### 4.2 水平拼接

```python
# 水平拼接
hstack = np.hstack((a, b))
print("\n水平拼接:")
print(hstack)
print("形状:", hstack.shape)
```

### 4.3 深度拼接

```python
# 深度拼接
dstack = np.dstack((a, b))
print("\n深度拼接:")
print(dstack)
print("形状:", dstack.shape)
```

### 4.4 通用拼接

```python
# 使用concatenate
print("\n使用concatenate垂直拼接:")
print(np.concatenate((a, b), axis=0))
print("\n使用concatenate水平拼接:")
print(np.concatenate((a, b), axis=1))
```

## 第五章：数组分割

### 5.1 垂直分割

```python
arr = np.arange(12).reshape(3, 4)
print("原始数组:")
print(arr)

# 垂直分割
vsplit = np.vsplit(arr, 3)
print("\n垂直分割:")
for i, part in enumerate(vsplit):
    print(f"部分 {i+1}:")
    print(part)
```

### 5.2 水平分割

```python
# 水平分割
hsplit = np.hsplit(arr, 2)
print("\n水平分割:")
for i, part in enumerate(hsplit):
    print(f"部分 {i+1}:")
    print(part)
```

### 5.3 通用分割

```python
# 使用split
print("\n使用split垂直分割:")
print(np.split(arr, 3, axis=0))
print("\n使用split水平分割:")
print(np.split(arr, 2, axis=1))
```

## 第六章：其他操作

### 6.1 重复

```python
arr = np.array([1, 2, 3])
print("原始数组:", arr)

# 重复元素
repeated = np.repeat(arr, 3)
print("\n重复元素:", repeated)

# 重复数组
tiled = np.tile(arr, 3)
print("\n重复数组:", tiled)
```

### 6.2 排序

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print("原始数组:", arr)

# 排序
sorted_arr = np.sort(arr)
print("\n排序后:", sorted_arr)

# 二维数组排序
arr_2d = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
print("\n二维数组:")
print(arr_2d)

# 按行排序
sorted_row = np.sort(arr_2d, axis=1)
print("\n按行排序:")
print(sorted_row)

# 按列排序
sorted_col = np.sort(arr_2d, axis=0)
print("\n按列排序:")
print(sorted_col)
```

### 6.3 唯一值

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
print("原始数组:", arr)

# 唯一值
unique = np.unique(arr)
print("\n唯一值:", unique)

# 唯一值及其计数
unique, counts = np.unique(arr, return_counts=True)
print("\n唯一值及其计数:")
print(dict(zip(unique, counts)))
```

## 第七章：性能优化

### 7.1 索引性能对比

| 索引类型 | 时间复杂度 | 相对速度 | 内存使用 |
|---------|-----------|---------|---------|
| 基本索引 | O(1) | 100% | 低 |
| 切片 | O(k) | 90% | 中 |
| 布尔索引 | O(n) | 75% | 高 |
| 花式索引 | O(k) | 65% | 高 |

### 7.2 内存管理

```python
# 视图 vs 副本
arr = np.arange(10)

# 切片创建视图
view = arr[::2]
print("视图:", view)

# 修改视图会影响原数组
view[0] = 99
print("修改视图后原数组:", arr)

# 花式索引创建副本
copy = arr[[0, 2, 4]]
print("副本:", copy)

# 修改副本不影响原数组
copy[0] = 100
print("修改副本后原数组:", arr)
```

### 7.3 最佳实践

1. **优先使用基本索引和切片**
2. **使用视图而非副本**
3. **避免频繁的数组拼接**
4. **预分配内存**
5. **使用向量化操作**
6. **合理使用reshape调整形状**

## 第八章：应用场景

### 8.1 数据提取

- **特征选择**：从数据集中提取特定特征
- **条件筛选**：根据条件筛选数据
- **数据采样**：随机或有选择地采样数据

### 8.2 数据转换

- **维度调整**：调整数据维度以适应模型
- **格式转换**：转换数据格式以满足需求
- **数据标准化**：对数据进行标准化处理

### 8.3 数据组合

- **特征融合**：将多个特征组合为一个
- **时间序列合并**：合并多个时间序列数据
- **数据集合并**：合并训练和测试数据

### 8.4 数据预处理

- **数据清洗**：处理缺失值和异常值
- **数据排序**：对数据进行排序
- **数据去重**：去除重复数据

## 第九章：习题

### 9.1 选择题

1. 以下哪个操作返回的是数组的视图？
   - A) arr.flatten()
   - B) arr.ravel()
   - C) arr.copy()
   - D) arr[[0, 1]]

2. 以下哪个操作用于垂直拼接数组？
   - A) np.hstack()
   - B) np.vstack()
   - C) np.dstack()
   - D) np.concatenate()

3. 以下哪种索引方式性能最快？
   - A) 基本索引
   - B) 切片
   - C) 布尔索引
   - D) 花式索引

### 9.2 填空题

1. 数组转置的快捷方式是使用________________。
2. 返回数组唯一值的函数是________________。
3. 自动计算数组维度时使用的符号是________________。

### 9.3 简答题

1. 简述视图和副本的区别。
2. 简述不同索引方式的性能差异。
3. 简述数组拼接的不同方法及其适用场景。

### 9.4 编程题

1. 创建一个5x5的随机数组，然后提取其对角线元素。
2. 将一个1D数组重塑为3x4的2D数组。
3. 创建两个2x2的数组，然后垂直和水平拼接它们。
4. 对一个1D数组进行排序，并返回排序后的结果。

## 第十章：总结

### 10.1 知识回顾

1. **索引和切片**：基本索引、切片、布尔索引、花式索引
2. **形状操作**：重塑、展平、转置
3. **拼接和分割**：垂直拼接、水平拼接、深度拼接、分割
4. **其他操作**：重复、排序、唯一值
5. **性能优化**：视图vs副本、索引性能、内存管理

### 10.2 学习建议

1. **实践练习**：多练习不同的数组操作
2. **性能测试**：比较不同操作的性能
3. **内存分析**：了解不同操作的内存使用
4. **应用开发**：在实际项目中应用数组操作

### 10.3 进阶学习

1. **数学运算**
2. **广播机制**
3. **线性代数**
4. **随机数生成**
5. **文件I/O操作**