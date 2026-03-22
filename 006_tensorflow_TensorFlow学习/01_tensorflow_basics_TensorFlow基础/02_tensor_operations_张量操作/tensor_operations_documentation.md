# TensorFlow 张量操作详细文档

## 1. 基本算术运算

### 1.1 加减乘除

```python
# 创建两个张量
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# 加法
add = tf.add(a, b)  # 或直接使用 a + b

# 减法
subtract = tf.subtract(a, b)  # 或直接使用 a - b

# 乘法
multiply = tf.multiply(a, b)  # 或直接使用 a * b

# 除法
divide = tf.divide(a, b)  # 或直接使用 a / b
```

### 1.2 其他算术运算

```python
# 取模
mod = tf.mod(a, b)  # 或直接使用 a % b

# 幂运算
power = tf.pow(a, b)  # 或直接使用 a ** b

# 平方
square = tf.square(a)

# 平方根
sqrt = tf.sqrt(tf.cast(a, tf.float32))
```

## 2. 矩阵运算

### 2.1 矩阵乘法

```python
# 创建两个矩阵
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])

# 矩阵乘法
matmul = tf.matmul(matrix1, matrix2)
# 或使用 @ 运算符
matmul = matrix1 @ matrix2
```

### 2.2 矩阵转置

```python
# 矩阵转置
transpose = tf.transpose(matrix1)
```

### 2.3 矩阵求逆

```python
# 矩阵求逆
inverse = tf.linalg.inv(matrix1)
```

### 2.4 矩阵行列式

```python
# 矩阵行列式
determinant = tf.linalg.det(matrix1)
```

## 3. 聚合运算

### 3.1 求和

```python
# 创建张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 求和
sum_all = tf.reduce_sum(tensor)

# 按行求和
sum_rows = tf.reduce_sum(tensor, axis=0)

# 按列求和
sum_cols = tf.reduce_sum(tensor, axis=1)
```

### 3.2 平均值

```python
# 平均值
mean_all = tf.reduce_mean(tensor)

# 按行求平均值
mean_rows = tf.reduce_mean(tensor, axis=0)

# 按列求平均值
mean_cols = tf.reduce_mean(tensor, axis=1)
```

### 3.3 最大值和最小值

```python
# 最大值
max_all = tf.reduce_max(tensor)

# 最小值
min_all = tf.reduce_min(tensor)
```

### 3.4 标准差和方差

```python
# 标准差
std_all = tf.math.reduce_std(tf.cast(tensor, tf.float32))

# 方差
variance_all = tf.math.reduce_variance(tf.cast(tensor, tf.float32))
```

## 4. 形状操作

### 4.1 重塑

```python
# 创建张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# 重塑为 [3, 2]
reshaped = tf.reshape(tensor, [3, 2])

# 展平为一维
flattened = tf.reshape(tensor, [-1])
```

### 4.2 维度操作

```python
# 增加维度
expanded = tf.expand_dims(tensor, axis=0)

# 减少维度
squeezed = tf.squeeze(expanded, axis=0)
```

### 4.3 转置

```python
# 转置
transposed = tf.transpose(tensor)
```

## 5. 逻辑运算

```python
# 创建布尔张量
a = tf.constant([True, False, True])
b = tf.constant([False, False, True])

# 逻辑与
logical_and = tf.logical_and(a, b)

# 逻辑或
logical_or = tf.logical_or(a, b)

# 逻辑非
logical_not = tf.logical_not(a)

# 逻辑异或
logical_xor = tf.logical_xor(a, b)
```

## 6. 比较运算

```python
# 创建张量
a = tf.constant([1, 2, 3, 4, 5])
b = tf.constant([3, 2, 1, 4, 6])

# 等于
equal = tf.equal(a, b)

# 不等于
not_equal = tf.not_equal(a, b)

# 大于
greater = tf.greater(a, b)

# 小于
less = tf.less(a, b)

# 大于等于
greater_equal = tf.greater_equal(a, b)

# 小于等于
less_equal = tf.less_equal(a, b)
```

## 7. 数学函数

```python
# 创建张量
tensor = tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0])

# 绝对值
abs_tensor = tf.abs(tensor)

# 指数
exp_tensor = tf.exp(tensor)

# 对数
log_tensor = tf.math.log(tf.abs(tensor) + 1e-10)

# 正弦
sin_tensor = tf.sin(tensor)

# 余弦
cos_tensor = tf.cos(tensor)

# 正切
tan_tensor = tf.tan(tensor)
```

## 8. 张量裁剪

```python
# 创建张量
tensor = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

# 裁剪到 [0, 2]
clipped = tf.clip_by_value(tensor, 0, 2)

# 裁剪到指定范数
norm_clipped = tf.clip_by_norm(tensor, 3)
```

## 9. 张量排序

```python
# 创建张量
tensor = tf.constant([3, 1, 4, 1, 5, 9, 2, 6])

# 排序
sorted_tensor = tf.sort(tensor)

# 降序排序
sorted_desc = tf.sort(tensor, direction='DESCENDING')

#  argsort
indices = tf.argsort(tensor)

# 顶部 k 个元素
top_k = tf.math.top_k(tensor, k=3)
```

## 10. 高级操作

### 10.1 广播

```python
# 创建张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 广播加法
broadcasted = tensor + tf.constant([1, 2, 3])
```

### 10.2 条件操作

```python
# 创建张量
condition = tf.constant([True, False, True])
x = tf.constant([1, 2, 3])
y = tf.constant([4, 5, 6])

# 条件操作
where = tf.where(condition, x, y)
```

### 10.3 唯一值

```python
# 创建张量
tensor = tf.constant([1, 2, 2, 3, 3, 3])

# 唯一值
unique = tf.unique(tensor)
```

## 11. 性能优化

- 使用 `tf.function` 装饰器将 Python 函数转换为 TensorFlow 计算图，提高执行效率。
- 尽量使用向量化操作，避免 Python 循环。
- 对于大型张量操作，考虑使用 `tf.TensorFlow` 的并行计算能力。
- 使用 `tf.data.Dataset` 进行高效的数据加载和预处理。

## 12. 常见问题和解决方案

### 12.1 类型不匹配
**问题**：不同类型的张量进行操作时出错。
**解决方案**：使用 `tf.cast()` 进行类型转换。

### 12.2 形状不匹配
**问题**：形状不同的张量进行操作时出错。
**解决方案**：使用 `tf.reshape()` 或 `tf.expand_dims()` 调整张量形状，或使用广播机制。

### 12.3 内存不足
**问题**：处理大型张量时内存不足。
**解决方案**：使用适当的批处理大小，或考虑使用 `tf.data.Dataset` 进行流式处理。

### 12.4 性能问题
**问题**：张量操作执行速度慢。
**解决方案**：使用 `tf.function` 装饰器，使用向量化操作，避免 Python 循环。