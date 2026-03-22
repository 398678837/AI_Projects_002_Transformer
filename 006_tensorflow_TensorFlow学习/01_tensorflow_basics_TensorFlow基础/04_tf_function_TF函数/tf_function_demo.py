#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow tf.function 演示

本脚本演示 TensorFlow tf.function 的使用方法和性能对比。
"""

import tensorflow as tf
import time

print("TensorFlow tf.function 演示")
print("=" * 50)

# 1. 基本使用
def basic_usage():
    print("\n1. 基本使用:")
    
    # 定义一个普通函数
    def add(a, b):
        return a + b
    
    # 使用 tf.function 装饰器
    @tf.function
    def tf_add(a, b):
        return a + b
    
    # 测试
    a = tf.constant(5)
    b = tf.constant(3)
    
    print(f"普通函数结果: {add(a, b)}")
    print(f"tf.function 结果: {tf_add(a, b)}")
    
    # 也可以不使用装饰器
    tf_add_alt = tf.function(add)
    print(f"tf.function 替代方法结果: {tf_add_alt(a, b)}")

# 2. 性能对比
def performance_comparison():
    print("\n2. 性能对比:")
    
    # 定义普通函数
    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    
    # 定义 tf.function 函数
    @tf.function
    def tf_fibonacci(n):
        a, b = tf.constant(0), tf.constant(1)
        for _ in tf.range(n):
            a, b = b, a + b
        return a
    
    # 测试性能
    n = 100000
    
    # 普通函数
    start = time.time()
    result = fibonacci(n)
    end = time.time()
    print(f"普通函数执行时间: {end - start:.4f} 秒")
    
    # tf.function 函数
    start = time.time()
    result = tf_fibonacci(n)
    end = time.time()
    print(f"tf.function 执行时间: {end - start:.4f} 秒")

# 3. 自动图转换
def auto_graph_conversion():
    print("\n3. 自动图转换:")
    
    @tf.function
    def square_if_positive(x):
        if x > 0:
            x = x * x
        else:
            x = x + 1
        return x
    
    # 查看生成的图
    print("生成的计算图:")
    print(tf.autograph.to_code(square_if_positive.python_function))
    
    # 测试
    print(f"square_if_positive(5): {square_if_positive(5)}")
    print(f"square_if_positive(-3): {square_if_positive(-3)}")

# 4. 变量和副作用
def variables_and_side_effects():
    print("\n4. 变量和副作用:")
    
    # 定义变量
    counter = tf.Variable(0)
    
    @tf.function
    def increment():
        counter.assign_add(1)
        return counter
    
    # 测试
    print(f"初始值: {increment()}")
    print(f"增加后: {increment()}")
    print(f"再增加: {increment()}")

# 5. 函数签名
def function_signatures():
    print("\n5. 函数签名:")
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add_one(x):
        return x + 1
    
    # 测试
    print(f"add_one(5.0): {add_one(5.0)}")
    
    # 尝试传入不同类型（应该失败）
    try:
        add_one(tf.constant(5, dtype=tf.int32))
    except Exception as e:
        print(f"类型不匹配错误: {e}")

# 6. 多态函数
def polymorphic_functions():
    print("\n6. 多态函数:")
    
    @tf.function
    def add(a, b):
        return a + b
    
    # 不同类型的输入
    print(f"add(1, 2): {add(1, 2)}")
    print(f"add(1.0, 2.0): {add(1.0, 2.0)}")
    print(f"add([1, 2], [3, 4]): {add([1, 2], [3, 4])}")
    
    # 查看函数的具体实现
    print(f"函数多态性: {add.pretty_printed_concrete_functions()}")

# 7. 控制流
def control_flow():
    print("\n7. 控制流:")
    
    @tf.function
    def for_loop(n):
        sum = tf.constant(0)
        for i in tf.range(n):
            sum += i
        return sum
    
    @tf.function
    def while_loop(n):
        sum = tf.constant(0)
        i = tf.constant(0)
        while i < n:
            sum += i
            i += 1
        return sum
    
    # 测试
    print(f"for_loop(5): {for_loop(5)}")
    print(f"while_loop(5): {while_loop(5)}")

# 8. 嵌套函数
def nested_functions():
    print("\n8. 嵌套函数:")
    
    @tf.function
    def outer_function(x):
        @tf.function
        def inner_function(y):
            return y * 2
        return inner_function(x) + x
    
    # 测试
    print(f"outer_function(5): {outer_function(5)}")

# 9. 输入输出规范
def input_output_specs():
    print("\n9. 输入输出规范:")
    
    @tf.function
    def multiply(a, b):
        return a * b
    
    # 获取函数的输入输出规范
    concrete_func = multiply.get_concrete_function(tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32))
    print(f"输入规范: {concrete_func.input_signature}")
    print(f"输出规范: {concrete_func.structured_outputs}")

# 10. 最佳实践
def best_practices():
    print("\n10. 最佳实践:")
    print("- 在性能关键的部分使用 tf.function")
    print("- 避免在 tf.function 内部使用 Python 原生数据结构，尽量使用 TensorFlow 张量")
    print("- 对于复杂的控制流，使用 TensorFlow 的控制流操作（如 tf.cond, tf.while_loop）")
    print("- 避免在 tf.function 内部创建变量，变量应该在函数外部定义")
    print("- 对于需要频繁调用的小函数，tf.function 可能不会带来性能提升")
    print("- 使用 @tf.function(input_signature=...) 来指定输入类型，避免不必要的多态")

if __name__ == "__main__":
    # 执行所有演示
    basic_usage()
    performance_comparison()
    auto_graph_conversion()
    variables_and_side_effects()
    function_signatures()
    polymorphic_functions()
    control_flow()
    nested_functions()
    input_output_specs()
    best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")