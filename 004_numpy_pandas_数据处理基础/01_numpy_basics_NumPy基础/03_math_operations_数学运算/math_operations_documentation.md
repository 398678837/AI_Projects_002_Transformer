# 数学运算详细文档

## 1. 什么是数学运算

数学运算是NumPy的核心功能之一，提供了丰富的数学函数和操作，用于对数组进行各种数值计算。NumPy的数学运算基于C语言实现，具有高性能和向量化特性。

### 1.1 核心优势

- **向量化操作**：无需循环，直接对整个数组进行运算
- **高性能**：基于C语言实现，运算速度快
- **丰富的函数**：提供大量数学函数
- **广播机制**：自动处理不同形状数组的运算
- **线性代数支持**：完整的线性代数运算

## 2. 基本算术运算

### 2.1 元素级运算

```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("加法:", a + b)      # [ 6,  8, 10, 12]
print("减法:", a - b)      # [-4, -4, -4, -4]
print("乘法:", a * b)      # [ 5, 12, 21, 32]
print("除法:", a / b)      # [0.2, 0.33333333, 0.42857143, 0.5]
print("取余:", a % b)      # [1, 2, 3, 4]
print("幂运算:", a ** b)    # [1, 64, 2187, 65536]
```

### 2.2 标量运算

```python
arr = np.array([1, 2, 3, 4])

print("加5:", arr + 5)      # [6, 7, 8, 9]
print("乘2:", arr * 2)      # [2, 4, 6, 8]
print("平方:", arr ** 2)    # [1, 4, 9, 16]
print("平方根:", np.sqrt(arr))  # [1.0, 1.41421356, 1.73205081, 2.0]
```

## 3. 三角函数

### 3.1 基本三角函数

```python
angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

print("正弦:", np.sin(angles))    # [ 0.00000000e+00,  1.00000000e+00,  1.22464680e-16, -1.00000000e+00, -2.44929360e-16]
print("余弦:", np.cos(angles))    # [ 1.00000000e+00,  6.12323400e-17, -1.00000000e+00, -1.83697020e-16,  1.00000000e+00]
print("正切:", np.tan(angles))    # [ 0.00000000e+00,  1.63312394e+16, -1.22464680e-16,  5.44374645e+15, -2.44929360e-16]
```

### 3.2 反三角函数

```python
values = np.array([0, 1, 0, -1, 0])

print("反正弦:", np.arcsin(values))  # [ 0.          1.57079633  0.         -1.57079633  0.        ]
print("反余弦:", np.arccos(values))  # [1.57079633 0.         1.57079633 3.14159265 1.57079633]
print("反正切:", np.arctan(values))  # [ 0.          0.78539816  0.         -0.78539816  0.        ]
```

## 4. 指数和对数

### 4.1 指数函数

```python
arr = np.array([1, 2, 3, 4])

print("指数:", np.exp(arr))      # [ 2.71828183,  7.3890561,  20.08553692,  54.59815003]
print("2的幂:", np.power(2, arr))  # [ 2,  4,  8, 16]
```

### 4.2 对数函数

```python
print("自然对数:", np.log(arr))      # [0.        , 0.69314718, 1.09861229, 1.38629436]
print("以10为底的对数:", np.log10(arr))  # [0.        , 0.30103   , 0.47712125, 0.60205999]
print("以2为底的对数:", np.log2(arr))   # [0.        , 1.        , 1.5849625 , 2.        ]
```

## 5. 统计运算

### 5.1 基本统计函数

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("总和:", np.sum(arr))            # 45
print("按行求和:", np.sum(arr, axis=0))   # [12, 15, 18]
print("按列求和:", np.sum(arr, axis=1))   # [ 6, 15, 24]

print("均值:", np.mean(arr))            # 5.0
print("按行求均值:", np.mean(arr, axis=0)) # [4. 5. 6.]
print("按列求均值:", np.mean(arr, axis=1)) # [2. 5. 8.]

print("标准差:", np.std(arr))           # 2.581988897471611
print("方差:", np.var(arr))            # 6.666666666666667
print("最小值:", np.min(arr))            # 1
print("最大值:", np.max(arr))            # 9
print("最小值索引:", np.argmin(arr))        # 0
print("最大值索引:", np.argmax(arr))        # 8
```

### 5.2 累积运算

```python
arr = np.array([1, 2, 3, 4, 5])

print("累积和:", np.cumsum(arr))  # [ 1,  3,  6, 10, 15]
print("累积积:", np.cumprod(arr)) # [  1,   2,   6,  24, 120]
```

## 6. 比较运算

### 6.1 元素级比较

```python
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 4, 4])

print("a == b:", a == b)  # [False,  True, False,  True]
print("a != b:", a != b)  # [ True, False,  True, False]
print("a < b:", a < b)    # [ True, False,  True, False]
print("a > b:", a > b)    # [False, False, False, False]
print("a <= b:", a <= b)  # [ True,  True,  True,  True]
print("a >= b:", a >= b)  # [False,  True, False,  True]
```

### 6.2 逻辑运算

```python
a = np.array([True, False, True, False])
b = np.array([True, True, False, False])

print("逻辑与:", np.logical_and(a, b))  # [ True, False, False, False]
print("逻辑或:", np.logical_or(a, b))   # [ True,  True,  True, False]
print("逻辑非:", np.logical_not(a))     # [False,  True, False,  True]
print("逻辑异或:", np.logical_xor(a, b)) # [False,  True,  True, False]
```

## 7. 其他数学函数

### 7.1 绝对值和符号

```python
arr = np.array([-1, 2, -3, 4, -5])

print("绝对值:", np.abs(arr))     # [1, 2, 3, 4, 5]
print("符号:", np.sign(arr))      # [-1,  1, -1,  1, -1]
```

### 7.2 取整函数

```python
arr = np.array([1.2, 2.7, -3.5, 4.9])

print("向上取整:", np.ceil(arr))   # [ 2.,  3., -3.,  5.]
print("向下取整:", np.floor(arr))  # [ 1.,  2., -4.,  4.]
print("四舍五入:", np.round(arr))  # [ 1.,  3., -4.,  5.]
print("截断:", np.trunc(arr))     # [ 1.,  2., -3.,  4.]
```

### 7.3 最值函数

```python
a = np.array([1, 3, 5])
b = np.array([2, 4, 6])

print("元素级最大值:", np.maximum(a, b))  # [2, 4, 6]
print("元素级最小值:", np.minimum(a, b))  # [1, 3, 5]
print("最大值与0的较大值:", np.maximum(a, 0))  # [1, 3, 5]
print("最小值与0的较小值:", np.minimum(a, 0))  # [0, 0, 0]
```

## 8. 线性代数运算

### 8.1 矩阵乘法

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("矩阵乘法:")
print(np.dot(a, b))
# 输出:
# [[19 22]
#  [43 50]]

# 或使用 @ 运算符
print("矩阵乘法 (@):")
print(a @ b)
```

### 8.2 矩阵转置

```python
print("矩阵转置:")
print(a.T)
# 输出:
# [[1 3]
#  [2 4]]

print("使用transpose:")
print(a.transpose())
```

### 8.3 矩阵行列式和逆

```python
print("矩阵行列式:", np.linalg.det(a))  # -2.0000000000000004

print("矩阵逆:")
print(np.linalg.inv(a))
# 输出:
# [[-2.   1. ]
#  [ 1.5 -0.5]]
```

### 8.4 特征值和特征向量

```python
print("特征值:", np.linalg.eigvals(a))  # [-0.37228132  5.37228132]

print("特征值和特征向量:")
values, vectors = np.linalg.eig(a)
print("特征值:", values)
print("特征向量:")
print(vectors)
```

## 9. 性能优化

### 9.1 向量化操作

```python
import time

# 生成大型数组
large_arr = np.random.rand(1000000)

# 测试NumPy向量化操作
start = time.time()
result_numpy = np.sin(large_arr) + np.cos(large_arr)
end = time.time()
print("NumPy向量化操作时间:", end - start, "秒")

# 测试Python循环
start = time.time()
result_loop = []
for x in large_arr:
    result_loop.append(np.sin(x) + np.cos(x))
end = time.time()
print("Python循环操作时间:", end - start, "秒")
```

### 9.2 数据类型选择

```python
# 使用float32代替float64节省内存
arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

print("float32内存占用:", arr_float32.nbytes, "字节")  # 12字节
print("float64内存占用:", arr_float64.nbytes, "字节")  # 24字节
```

### 9.3 避免中间数组

```python
# 避免创建中间数组
arr = np.random.rand(1000000)

# 不好的做法
start = time.time()
temp = np.sin(arr)
result = temp + np.cos(arr)
end = time.time()
print("使用中间数组时间:", end - start, "秒")

# 好的做法
start = time.time()
result = np.sin(arr) + np.cos(arr)
end = time.time()
print("直接计算时间:", end - start, "秒")
```

## 10. 应用场景

### 10.1 数据处理

- **数值计算**：科学计算、工程计算
- **统计分析**：数据分析、统计建模
- **信号处理**：信号滤波、傅里叶变换

### 10.2 机器学习

- **特征工程**：特征提取、特征变换
- **模型训练**：梯度计算、损失函数
- **模型评估**：性能指标计算

### 10.3 图像处理

- **图像变换**：缩放、旋转、滤波
- **像素操作**：亮度调整、对比度调整
- **特征提取**：边缘检测、特征点提取

### 10.4 金融分析

- **风险评估**：波动率计算、风险价值
- **投资组合**：资产配置、优化
- **时间序列**：趋势分析、预测

## 11. 最佳实践

### 11.1 性能优化

1. **使用向量化操作**：避免Python循环
2. **选择合适的数据类型**：根据需求选择数据类型
3. **避免中间数组**：直接进行复合运算
4. **使用就地操作**：减少内存使用
5. **利用广播机制**：简化代码

### 11.2 代码可读性

1. **使用有意义的变量名**：提高代码可读性
2. **添加注释**：解释复杂的数学运算
3. **模块化**：将复杂运算封装为函数
4. **使用NumPy函数**：避免重复造轮子

### 11.3 常见错误

1. **整数除法**：Python 2中的整数除法问题
2. **精度问题**：浮点数精度误差
3. **形状不匹配**：运算时数组形状不匹配
4. **内存不足**：处理大型数组时内存不足

## 12. 总结

NumPy的数学运算功能强大且高效，是科学计算和数据处理的重要工具。通过掌握各种数学运算函数和优化技巧，可以显著提高代码的性能和可读性。

### 12.1 核心要点

- **丰富的数学函数**：从基本运算到高级数学函数
- **向量化操作**：高性能计算
- **广播机制**：灵活处理不同形状的数组
- **线性代数支持**：完整的矩阵运算
- **统计功能**：数据分析必备

### 12.2 下一步学习

- 广播机制
- 线性代数高级应用
- 随机数生成
- 文件I/O操作