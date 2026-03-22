# TensorFlow 张量创建详细文档

## 1. 张量的基本概念

张量是 TensorFlow 中最基本的数据结构，是一个多维数组或列表。张量的维度称为阶（rank），例如：
- 标量（0阶张量）：单个数值
- 向量（1阶张量）：一维数组
- 矩阵（2阶张量）：二维数组
- 更高阶张量：三维及以上的数组

## 2. 张量的创建方法

### 2.1 从 Python 对象创建

```python
# 从列表创建
list_tensor = tf.constant([1, 2, 3, 4, 5])

# 从嵌套列表创建
nested_list_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# 从元组创建
tuple_tensor = tf.constant((1, 2, 3))

# 从 numpy 数组创建
import numpy as np
numpy_array = np.array([1, 2, 3, 4, 5])
numpy_tensor = tf.constant(numpy_array)
```

### 2.2 创建特殊张量

```python
# 创建全零张量
zeros_tensor = tf.zeros([2, 3])

# 创建全一张量
ones_tensor = tf.ones([3, 2])

# 创建常数张量
fill_tensor = tf.fill([2, 2], 7)

# 创建单位矩阵
eye_tensor = tf.eye(3)
```

### 2.3 创建随机张量

```python
# 创建均匀分布随机张量
uniform_tensor = tf.random.uniform([2, 3], minval=0, maxval=1)

# 创建正态分布随机张量
normal_tensor = tf.random.normal([2, 3], mean=0, stddev=1)

# 创建截断正态分布随机张量
truncated_normal_tensor = tf.random.truncated_normal([2, 3], mean=0, stddev=1)

# 创建随机打乱的张量
shuffled_tensor = tf.random.shuffle(tf.range(10))
```

### 2.4 创建序列张量

```python
# 创建从 0 到 9 的序列
range_tensor = tf.range(10)

# 创建从 2 到 10，步长为 2 的序列
range_step_tensor = tf.range(2, 10, 2)

# 创建等间隔序列
linspace_tensor = tf.linspace(0.0, 1.0, 5)
```

## 3. 张量的属性

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"张量形状: {tensor.shape}")
print(f"张量维度: {tensor.ndim}")
print(f"张量数据类型: {tensor.dtype}")
print(f"张量元素数量: {tf.size(tensor).numpy()}")
```

## 4. 张量的类型转换

```python
# 创建整数张量
int_tensor = tf.constant([1, 2, 3])

# 转换为浮点数张量
float_tensor = tf.cast(int_tensor, tf.float32)

# 转换为布尔张量
bool_tensor = tf.cast(int_tensor, tf.bool)
```

## 5. 张量的形状操作

```python
# 创建张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# 重塑张量
reshaped_tensor = tf.reshape(tensor, [3, 2])

# 展平张量
flattened_tensor = tf.reshape(tensor, [-1])

# 增加维度
expanded_tensor = tf.expand_dims(tensor, axis=0)

# 减少维度
squeezed_tensor = tf.squeeze(expanded_tensor, axis=0)
```

## 6. 张量的索引和切片

```python
# 创建张量
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 访问单个元素
print(tensor[0, 0].numpy())
print(tensor[1, 2].numpy())

# 访问整行
print(tensor[0, :].numpy())
print(tensor[1, :].numpy())

# 访问整列
print(tensor[:, 0].numpy())
print(tensor[:, 1].numpy())

# 切片操作
print(tensor[0:2, 1:3].numpy())
print(tensor[1:, :2].numpy())
```

## 7. 张量的拼接和拆分

```python
# 创建两个张量
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

# 沿轴 0 拼接
concat_axis0 = tf.concat([tensor1, tensor2], axis=0)

# 沿轴 1 拼接
concat_axis1 = tf.concat([tensor1, tensor2], axis=1)

# 堆叠操作
stack = tf.stack([tensor1, tensor2])

# 拆分为两个张量
split = tf.split(concat_axis0, num_or_size_splits=2, axis=0)
```

## 8. 常量和变量

```python
# 创建常量
constant = tf.constant([1, 2, 3])

# 创建变量
variable = tf.Variable([1, 2, 3])

# 修改变量值
variable.assign([4, 5, 6])

# 变量自增
variable.assign_add([1, 1, 1])

# 变量自减
variable.assign_sub([1, 1, 1])
```

## 9. 常见问题和解决方案

### 9.1 张量类型不匹配
**问题**：操作两个不同类型的张量时出错。
**解决方案**：使用 `tf.cast()` 进行类型转换。

### 9.2 张量形状不匹配
**问题**：操作两个形状不同的张量时出错。
**解决方案**：使用 `tf.reshape()` 或 `tf.expand_dims()` 调整张量形状，或使用广播机制。

### 9.3 内存不足
**问题**：创建大型张量时内存不足。
**解决方案**：使用适当的批处理大小，或考虑使用 `tf.data.Dataset` 进行流式处理。

## 10. 最佳实践

- 使用 `tf.constant()` 创建不可变张量，使用 `tf.Variable()` 创建可变张量。
- 尽量使用 TensorFlow 原生函数创建张量，而不是从 Python 对象转换。
- 对于大型张量，考虑使用 `tf.TensorSpec` 或 `tf.SparseTensor` 以节省内存。
- 在计算图中，尽量使用张量操作而不是 Python 循环，以提高性能。
- 使用 `tf.debugging.assert_shapes()` 检查张量形状，避免运行时错误。