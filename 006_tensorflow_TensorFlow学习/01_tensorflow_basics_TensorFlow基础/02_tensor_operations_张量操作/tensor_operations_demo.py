#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow 张量操作演示

本脚本演示 TensorFlow 中张量的各种操作方法。
"""

import tensorflow as tf
import numpy as np

print("TensorFlow 张量操作演示")
print("=" * 50)

# 1. 基本算术运算
def basic_arithmetic_operations():
    print("\n1. 基本算术运算:")
    
    # 创建两个张量
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    print(f"张量 a: {a}")
    print(f"张量 b: {b}")
    
    # 加法
    add = tf.add(a, b)
    print(f"加法: {add}")
    
    # 减法
    subtract = tf.subtract(a, b)
    print(f"减法: {subtract}")
    
    # 乘法
    multiply = tf.multiply(a, b)
    print(f"乘法: {multiply}")
    
    # 除法
    divide = tf.divide(a, b)
    print(f"除法: {divide}")
    
    # 取模
    mod = tf.mod(a, b)
    print(f"取模: {mod}")
    
    # 幂运算
    power = tf.pow(a, b)
    print(f"幂运算: {power}")

# 2. 矩阵运算
def matrix_operations():
    print("\n2. 矩阵运算:")
    
    # 创建两个矩阵
    matrix1 = tf.constant([[1, 2], [3, 4]])
    matrix2 = tf.constant([[5, 6], [7, 8]])
    print(f"矩阵 1:\n{matrix1}")
    print(f"矩阵 2:\n{matrix2}")
    
    # 矩阵乘法
    matmul = tf.matmul(matrix1, matrix2)
    print(f"矩阵乘法:\n{matmul}")
    
    # 矩阵转置
    transpose = tf.transpose(matrix1)
    print(f"矩阵转置:\n{transpose}")
    
    # 矩阵求逆
    inverse = tf.linalg.inv(matrix1)
    print(f"矩阵求逆:\n{inverse}")
    
    # 矩阵行列式
    determinant = tf.linalg.det(matrix1)
    print(f"矩阵行列式: {determinant}")
    
    # 矩阵迹
    trace = tf.linalg.trace(matrix1)
    print(f"矩阵迹: {trace}")

# 3. 聚合运算
def aggregation_operations():
    print("\n3. 聚合运算:")
    
    # 创建张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始张量:\n{tensor}")
    
    # 求和
    sum_all = tf.reduce_sum(tensor)
    print(f"总和: {sum_all}")
    
    # 按行求和
    sum_rows = tf.reduce_sum(tensor, axis=0)
    print(f"按行求和: {sum_rows}")
    
    # 按列求和
    sum_cols = tf.reduce_sum(tensor, axis=1)
    print(f"按列求和: {sum_cols}")
    
    # 平均值
    mean_all = tf.reduce_mean(tensor)
    print(f"平均值: {mean_all}")
    
    # 最大值
    max_all = tf.reduce_max(tensor)
    print(f"最大值: {max_all}")
    
    # 最小值
    min_all = tf.reduce_min(tensor)
    print(f"最小值: {min_all}")
    
    # 标准差
    std_all = tf.math.reduce_std(tf.cast(tensor, tf.float32))
    print(f"标准差: {std_all}")

# 4. 形状操作
def shape_operations():
    print("\n4. 形状操作:")
    
    # 创建张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"原始张量: {tensor}, 形状: {tensor.shape}")
    
    # 重塑
    reshaped = tf.reshape(tensor, [3, 2])
    print(f"重塑为 [3, 2]: {reshaped}, 形状: {reshaped.shape}")
    
    # 展平
    flattened = tf.reshape(tensor, [-1])
    print(f"展平: {flattened}, 形状: {flattened.shape}")
    
    # 增加维度
    expanded = tf.expand_dims(tensor, axis=0)
    print(f"增加维度: {expanded}, 形状: {expanded.shape}")
    
    # 减少维度
    squeezed = tf.squeeze(expanded, axis=0)
    print(f"减少维度: {squeezed}, 形状: {squeezed.shape}")
    
    # 转置
    transposed = tf.transpose(tensor)
    print(f"转置: {transposed}, 形状: {transposed.shape}")

# 5. 逻辑运算
def logical_operations():
    print("\n5. 逻辑运算:")
    
    # 创建布尔张量
    a = tf.constant([True, False, True])
    b = tf.constant([False, False, True])
    print(f"张量 a: {a}")
    print(f"张量 b: {b}")
    
    # 逻辑与
    logical_and = tf.logical_and(a, b)
    print(f"逻辑与: {logical_and}")
    
    # 逻辑或
    logical_or = tf.logical_or(a, b)
    print(f"逻辑或: {logical_or}")
    
    # 逻辑非
    logical_not = tf.logical_not(a)
    print(f"逻辑非: {logical_not}")
    
    # 逻辑异或
    logical_xor = tf.logical_xor(a, b)
    print(f"逻辑异或: {logical_xor}")

# 6. 比较运算
def comparison_operations():
    print("\n6. 比较运算:")
    
    # 创建张量
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([3, 2, 1, 4, 6])
    print(f"张量 a: {a}")
    print(f"张量 b: {b}")
    
    # 等于
    equal = tf.equal(a, b)
    print(f"等于: {equal}")
    
    # 不等于
    not_equal = tf.not_equal(a, b)
    print(f"不等于: {not_equal}")
    
    # 大于
    greater = tf.greater(a, b)
    print(f"大于: {greater}")
    
    # 小于
    less = tf.less(a, b)
    print(f"小于: {less}")
    
    # 大于等于
    greater_equal = tf.greater_equal(a, b)
    print(f"大于等于: {greater_equal}")
    
    # 小于等于
    less_equal = tf.less_equal(a, b)
    print(f"小于等于: {less_equal}")

# 7. 数学函数
def mathematical_functions():
    print("\n7. 数学函数:")
    
    # 创建张量
    tensor = tf.constant([-1.0, 0.0, 1.0, 2.0, 3.0])
    print(f"原始张量: {tensor}")
    
    # 绝对值
    abs_tensor = tf.abs(tensor)
    print(f"绝对值: {abs_tensor}")
    
    # 平方根
    sqrt_tensor = tf.sqrt(tf.abs(tensor))
    print(f"平方根: {sqrt_tensor}")
    
    # 指数
    exp_tensor = tf.exp(tensor)
    print(f"指数: {exp_tensor}")
    
    # 对数
    log_tensor = tf.math.log(tf.abs(tensor) + 1e-10)
    print(f"对数: {log_tensor}")
    
    # 正弦
    sin_tensor = tf.sin(tensor)
    print(f"正弦: {sin_tensor}")
    
    # 余弦
    cos_tensor = tf.cos(tensor)
    print(f"余弦: {cos_tensor}")
    
    # 正切
    tan_tensor = tf.tan(tensor)
    print(f"正切: {tan_tensor}")

# 8. 张量裁剪
def tensor_clipping():
    print("\n8. 张量裁剪:")
    
    # 创建张量
    tensor = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    print(f"原始张量: {tensor}")
    
    # 裁剪到 [0, 2]
    clipped = tf.clip_by_value(tensor, 0, 2)
    print(f"裁剪到 [0, 2]: {clipped}")
    
    # 裁剪到指定范数
    norm_clipped = tf.clip_by_norm(tensor, 3)
    print(f"裁剪到范数 3: {norm_clipped}")

# 9. 张量排序
def tensor_sorting():
    print("\n9. 张量排序:")
    
    # 创建张量
    tensor = tf.constant([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"原始张量: {tensor}")
    
    # 排序
    sorted_tensor = tf.sort(tensor)
    print(f"排序后: {sorted_tensor}")
    
    # 降序排序
    sorted_desc = tf.sort(tensor, direction='DESCENDING')
    print(f"降序排序: {sorted_desc}")
    
    #  argsort
    indices = tf.argsort(tensor)
    print(f"排序索引: {indices}")
    
    # 顶部 k 个元素
    top_k = tf.math.top_k(tensor, k=3)
    print(f"顶部 3 个元素: {top_k.values}")
    print(f"顶部 3 个元素的索引: {top_k.indices}")

# 10. 高级操作
def advanced_operations():
    print("\n10. 高级操作:")
    
    # 创建张量
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始张量:\n{tensor}")
    
    # 广播
    broadcasted = tensor + tf.constant([1, 2, 3])
    print(f"广播加法:\n{broadcasted}")
    
    # 条件操作
    condition = tf.constant([True, False, True])
    x = tf.constant([1, 2, 3])
    y = tf.constant([4, 5, 6])
    where = tf.where(condition, x, y)
    print(f"条件操作: {where}")
    
    # 唯一值
    unique = tf.unique(tf.constant([1, 2, 2, 3, 3, 3]))
    print(f"唯一值: {unique.values}")
    print(f"唯一值索引: {unique.idx}")

if __name__ == "__main__":
    # 执行所有演示
    basic_arithmetic_operations()
    matrix_operations()
    aggregation_operations()
    shape_operations()
    logical_operations()
    comparison_operations()
    mathematical_functions()
    tensor_clipping()
    tensor_sorting()
    advanced_operations()
    
    print("\n" + "=" * 50)
    print("演示完成！")