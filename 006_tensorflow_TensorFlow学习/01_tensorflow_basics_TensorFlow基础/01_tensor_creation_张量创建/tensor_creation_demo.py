#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow 张量创建演示

本脚本演示 TensorFlow 中张量的创建方法和基本操作。
"""

import tensorflow as tf
import numpy as np

print("TensorFlow 张量创建演示")
print("=" * 50)

# 1. 从 Python 对象创建张量
def create_from_python_objects():
    print("\n1. 从 Python 对象创建张量:")
    
    # 从列表创建
    list_tensor = tf.constant([1, 2, 3, 4, 5])
    print(f"从列表创建: {list_tensor}")
    
    # 从嵌套列表创建
    nested_list_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"从嵌套列表创建: {nested_list_tensor}")
    
    # 从元组创建
    tuple_tensor = tf.constant((1, 2, 3))
    print(f"从元组创建: {tuple_tensor}")
    
    # 从 numpy 数组创建
    numpy_array = np.array([1, 2, 3, 4, 5])
    numpy_tensor = tf.constant(numpy_array)
    print(f"从 numpy 数组创建: {numpy_tensor}")

# 2. 创建特殊张量
def create_special_tensors():
    print("\n2. 创建特殊张量:")
    
    # 创建全零张量
    zeros_tensor = tf.zeros([2, 3])
    print(f"全零张量: {zeros_tensor}")
    
    # 创建全一张量
    ones_tensor = tf.ones([3, 2])
    print(f"全一张量: {ones_tensor}")
    
    # 创建常数张量
    fill_tensor = tf.fill([2, 2], 7)
    print(f"常数张量: {fill_tensor}")
    
    # 创建单位矩阵
    eye_tensor = tf.eye(3)
    print(f"单位矩阵: {eye_tensor}")

# 3. 创建随机张量
def create_random_tensors():
    print("\n3. 创建随机张量:")
    
    # 创建均匀分布随机张量
    uniform_tensor = tf.random.uniform([2, 3], minval=0, maxval=1)
    print(f"均匀分布随机张量: {uniform_tensor}")
    
    # 创建正态分布随机张量
    normal_tensor = tf.random.normal([2, 3], mean=0, stddev=1)
    print(f"正态分布随机张量: {normal_tensor}")
    
    # 创建截断正态分布随机张量
    truncated_normal_tensor = tf.random.truncated_normal([2, 3], mean=0, stddev=1)
    print(f"截断正态分布随机张量: {truncated_normal_tensor}")
    
    # 创建随机打乱的张量
    shuffled_tensor = tf.random.shuffle(tf.range(10))
    print(f"随机打乱的张量: {shuffled_tensor}")

# 4. 创建序列张量
def create_sequence_tensors():
    print("\n4. 创建序列张量:")
    
    # 创建从 0 到 9 的序列
    range_tensor = tf.range(10)
    print(f"从 0 到 9 的序列: {range_tensor}")
    
    # 创建从 2 到 10，步长为 2 的序列
    range_step_tensor = tf.range(2, 10, 2)
    print(f"从 2 到 10，步长为 2 的序列: {range_step_tensor}")
    
    # 创建等间隔序列
    linspace_tensor = tf.linspace(0.0, 1.0, 5)
    print(f"从 0.0 到 1.0 的 5 个等间隔值: {linspace_tensor}")

# 5. 张量属性
def tensor_attributes():
    print("\n5. 张量属性:")
    
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"张量: {tensor}")
    print(f"张量形状: {tensor.shape}")
    print(f"张量维度: {tensor.ndim}")
    print(f"张量数据类型: {tensor.dtype}")
    print(f"张量元素数量: {tf.size(tensor).numpy()}")

# 6. 张量类型转换
def tensor_type_conversion():
    print("\n6. 张量类型转换:")
    
    # 创建整数张量
    int_tensor = tf.constant([1, 2, 3])
    print(f"整数张量: {int_tensor}, 类型: {int_tensor.dtype}")
    
    # 转换为浮点数张量
    float_tensor = tf.cast(int_tensor, tf.float32)
    print(f"浮点数张量: {float_tensor}, 类型: {float_tensor.dtype}")
    
    # 转换为布尔张量
    bool_tensor = tf.cast(int_tensor, tf.bool)
    print(f"布尔张量: {bool_tensor}, 类型: {bool_tensor.dtype}")

# 7. 张量形状操作
def tensor_shape_operations():
    print("\n7. 张量形状操作:")
    
    # 创建张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"原始张量: {tensor}, 形状: {tensor.shape}")
    
    # 重塑张量
    reshaped_tensor = tf.reshape(tensor, [3, 2])
    print(f"重塑为 [3, 2]: {reshaped_tensor}, 形状: {reshaped_tensor.shape}")
    
    # 展平张量
    flattened_tensor = tf.reshape(tensor, [-1])
    print(f"展平张量: {flattened_tensor}, 形状: {flattened_tensor.shape}")
    
    # 增加维度
    expanded_tensor = tf.expand_dims(tensor, axis=0)
    print(f"增加维度: {expanded_tensor}, 形状: {expanded_tensor.shape}")
    
    # 减少维度
    squeezed_tensor = tf.squeeze(expanded_tensor, axis=0)
    print(f"减少维度: {squeezed_tensor}, 形状: {squeezed_tensor.shape}")

# 8. 张量索引和切片
def tensor_indexing_slicing():
    print("\n8. 张量索引和切片:")
    
    # 创建张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始张量:\n{tensor}")
    
    # 访问单个元素
    print(f"访问 (0, 0) 位置的元素: {tensor[0, 0].numpy()}")
    print(f"访问 (1, 2) 位置的元素: {tensor[1, 2].numpy()}")
    
    # 访问整行
    print(f"访问第一行: {tensor[0, :].numpy()}")
    print(f"访问第二行: {tensor[1, :].numpy()}")
    
    # 访问整列
    print(f"访问第一列: {tensor[:, 0].numpy()}")
    print(f"访问第二列: {tensor[:, 1].numpy()}")
    
    # 切片操作
    print(f"切片 [0:2, 1:3]:\n{tensor[0:2, 1:3].numpy()}")
    print(f"切片 [1:, :2]:\n{tensor[1:, :2].numpy()}")

# 9. 张量拼接和拆分
def tensor_concatenation_splitting():
    print("\n9. 张量拼接和拆分:")
    
    # 创建两个张量
    tensor1 = tf.constant([[1, 2], [3, 4]])
    tensor2 = tf.constant([[5, 6], [7, 8]])
    print(f"张量 1:\n{tensor1}")
    print(f"张量 2:\n{tensor2}")
    
    # 沿轴 0 拼接
    concat_axis0 = tf.concat([tensor1, tensor2], axis=0)
    print(f"沿轴 0 拼接:\n{concat_axis0}")
    
    # 沿轴 1 拼接
    concat_axis1 = tf.concat([tensor1, tensor2], axis=1)
    print(f"沿轴 1 拼接:\n{concat_axis1}")
    
    # 堆叠操作
    stack = tf.stack([tensor1, tensor2])
    print(f"堆叠操作:\n{stack}, 形状: {stack.shape}")
    
    # 拆分为两个张量
    split = tf.split(concat_axis0, num_or_size_splits=2, axis=0)
    print(f"拆分为两个张量:")
    print(f"第一个张量:\n{split[0]}")
    print(f"第二个张量:\n{split[1]}")

# 10. 常量和变量
def constants_and_variables():
    print("\n10. 常量和变量:")
    
    # 创建常量
    constant = tf.constant([1, 2, 3])
    print(f"常量: {constant}")
    
    # 创建变量
    variable = tf.Variable([1, 2, 3])
    print(f"变量: {variable}")
    
    # 修改变量值
    variable.assign([4, 5, 6])
    print(f"修改后变量: {variable}")
    
    # 变量自增
    variable.assign_add([1, 1, 1])
    print(f"自增后变量: {variable}")
    
    # 变量自减
    variable.assign_sub([1, 1, 1])
    print(f"自减后变量: {variable}")

if __name__ == "__main__":
    # 执行所有演示
    create_from_python_objects()
    create_special_tensors()
    create_random_tensors()
    create_sequence_tensors()
    tensor_attributes()
    tensor_type_conversion()
    tensor_shape_operations()
    tensor_indexing_slicing()
    tensor_concatenation_splitting()
    constants_and_variables()
    
    print("\n" + "=" * 50)
    print("演示完成！")