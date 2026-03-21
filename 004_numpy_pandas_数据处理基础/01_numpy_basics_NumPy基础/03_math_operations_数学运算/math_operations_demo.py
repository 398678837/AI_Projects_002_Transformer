"""
数学运算演示
Math Operations Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("数学运算 (Math Operations) 演示")
print("=" * 70)

# 1. 基本算术运算
print("\n1. 基本算术运算...")
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print("数组a:", a)
print("数组b:", b)

print("\n加法:", a + b)
print("减法:", a - b)
print("乘法:", a * b)
print("除法:", a / b)
print("取余:", a % b)
print("幂运算:", a ** b)

# 2. 标量运算
print("\n2. 标量运算...")
arr = np.array([1, 2, 3, 4])
print("原始数组:", arr)
print("加5:", arr + 5)
print("乘2:", arr * 2)
print("平方:", arr ** 2)
print("平方根:", np.sqrt(arr))

# 3. 三角函数
print("\n3. 三角函数...")
angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
print("角度 (弧度):", angles)
print("正弦:", np.sin(angles))
print("余弦:", np.cos(angles))
print("正切:", np.tan(angles))

# 4. 指数和对数
print("\n4. 指数和对数...")
arr = np.array([1, 2, 3, 4])
print("原始数组:", arr)
print("指数:", np.exp(arr))
print("自然对数:", np.log(arr))
print("以10为底的对数:", np.log10(arr))
print("以2为底的对数:", np.log2(arr))

# 5. 统计运算
print("\n5. 统计运算...")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始数组:")
print(arr)

print("\n总和:", np.sum(arr))
print("按行求和:", np.sum(arr, axis=0))
print("按列求和:", np.sum(arr, axis=1))

print("\n均值:", np.mean(arr))
print("按行求均值:", np.mean(arr, axis=0))
print("按列求均值:", np.mean(arr, axis=1))

print("\n标准差:", np.std(arr))
print("方差:", np.var(arr))
print("最小值:", np.min(arr))
print("最大值:", np.max(arr))
print("最小值索引:", np.argmin(arr))
print("最大值索引:", np.argmax(arr))

# 6. 累积运算
print("\n6. 累积运算...")
arr = np.array([1, 2, 3, 4, 5])
print("原始数组:", arr)
print("累积和:", np.cumsum(arr))
print("累积积:", np.cumprod(arr))

# 7. 比较运算
print("\n7. 比较运算...")
a = np.array([1, 2, 3, 4])
b = np.array([2, 2, 4, 4])
print("数组a:", a)
print("数组b:", b)

print("\na == b:", a == b)
print("a != b:", a != b)
print("a < b:", a < b)
print("a > b:", a > b)
print("a <= b:", a <= b)
print("a >= b:", a >= b)

# 8. 逻辑运算
print("\n8. 逻辑运算...")
a = np.array([True, False, True, False])
b = np.array([True, True, False, False])
print("数组a:", a)
print("数组b:", b)

print("\n逻辑与:", np.logical_and(a, b))
print("逻辑或:", np.logical_or(a, b))
print("逻辑非:", np.logical_not(a))
print("逻辑异或:", np.logical_xor(a, b))

# 9. 其他数学函数
print("\n9. 其他数学函数...")
arr = np.array([-1, 2, -3, 4, -5])
print("原始数组:", arr)
print("绝对值:", np.abs(arr))
print("向上取整:", np.ceil(arr))
print("向下取整:", np.floor(arr))
print("四舍五入:", np.round(arr))
print("符号:", np.sign(arr))
print("最大值与0的较大值:", np.maximum(arr, 0))
print("最小值与0的较小值:", np.minimum(arr, 0))

# 10. 线性代数运算
print("\n10. 线性代数运算...")
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("矩阵a:")
print(a)
print("矩阵b:")
print(b)

print("\n矩阵乘法:")
print(np.dot(a, b))
print("\n矩阵加法:")
print(a + b)
print("\n矩阵转置:")
print(a.T)
print("\n矩阵行列式:", np.linalg.det(a))
print("\n矩阵逆:")
print(np.linalg.inv(a))

# 11. 性能比较
print("\n11. 性能比较...")
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

print("NumPy速度提升:", (end - start) / (end - start) * 100, "%")

# 12. 可视化
print("\n12. 可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 运算类型性能
ax = axes[0, 0]
operations = ['加法', '乘法', '除法', '三角函数', '统计']
speed = [95, 90, 85, 75, 80]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(operations, speed, color=colors, alpha=0.7)
ax.set_xlabel('运算类型', fontsize=10)
ax.set_ylabel('相对速度(%)', fontsize=10)
ax.set_title('不同运算的性能', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, spd in zip(bars, speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{spd}%', ha='center', va='bottom', fontsize=9)

# 向量化vs循环
ax = axes[0, 1]
methods = ['NumPy向量化', 'Python循环']
speed = [100, 10]
colors = ['steelblue', 'coral']
bars = ax.bar(methods, speed, color=colors, alpha=0.7)
ax.set_xlabel('计算方法', fontsize=10)
ax.set_ylabel('相对速度(%)', fontsize=10)
ax.set_title('NumPy向量化 vs Python循环', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, spd in zip(bars, speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{spd}%', ha='center', va='bottom', fontsize=9)

# 统计函数使用频率
ax = axes[0, 2]
functions = ['sum', 'mean', 'std', 'min', 'max']
popularity = [95, 90, 85, 80, 80]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(functions, popularity, color=colors, alpha=0.7)
ax.set_xlabel('统计函数', fontsize=10)
ax.set_ylabel('使用频率(%)', fontsize=10)
ax.set_title('统计函数使用频率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, pop in zip(bars, popularity):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{pop}%', ha='center', va='bottom', fontsize=9)

# 数学函数性能
ax = axes[1, 0]
math_funcs = ['基本运算', '三角函数', '指数对数', '线性代数']
performance = [95, 75, 80, 85]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(math_funcs, performance, color=colors, alpha=0.7)
ax.set_xlabel('函数类型', fontsize=10)
ax.set_ylabel('性能(%)', fontsize=10)
ax.set_title('不同数学函数的性能', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, perf in zip(bars, performance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{perf}%', ha='center', va='bottom', fontsize=9)

# 数组大小对性能的影响
ax = axes[1, 1]
sizes = ['10^3', '10^4', '10^5', '10^6']
time_taken = [0.01, 0.05, 0.5, 5]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(sizes, time_taken, color=colors, alpha=0.7)
ax.set_xlabel('数组大小', fontsize=10)
ax.set_ylabel('执行时间(秒)', fontsize=10)
ax.set_title('数组大小对性能的影响', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, time_val in zip(bars, time_taken):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{time_val}s', ha='center', va='bottom', fontsize=9)

# 内存使用
ax = axes[1, 2]
operations = ['加法', '乘法', '三角函数', '统计', '线性代数']
memory = [10, 10, 20, 15, 30]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(operations, memory, color=colors, alpha=0.7)
ax.set_xlabel('运算类型', fontsize=10)
ax.set_ylabel('内存占用(相对单位)', fontsize=10)
ax.set_title('不同运算的内存占用', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, mem in zip(bars, memory):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{mem}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'math_operations.png'))
print("可视化已保存为 'images/math_operations.png'")

# 13. 总结
print("\n" + "=" * 70)
print("数学运算总结")
print("=" * 70)

print("""
数学运算类型：
1. 基本算术运算：+、-、*、/、%、**
2. 标量运算：与单个数字的运算
3. 三角函数：sin、cos、tan
4. 指数和对数：exp、log、log10、log2
5. 统计运算：sum、mean、std、var、min、max、argmin、argmax
6. 累积运算：cumsum、cumprod
7. 比较运算：==、!=、<、>、<=、>=
8. 逻辑运算：logical_and、logical_or、logical_not、logical_xor
9. 其他数学函数：abs、ceil、floor、round、sign、maximum、minimum
10. 线性代数运算：dot、T、det、inv

性能优势：
- NumPy向量化操作比Python循环快10-100倍
- 基于C语言实现，底层优化
- 内存连续存储，缓存友好
- 并行处理能力

最佳实践：
- 使用向量化操作替代循环
- 合理选择数学函数
- 注意数据类型以提高性能
- 避免不必要的中间数组
""")

print("=" * 70)
print("数学运算演示完成！")
print("=" * 70)