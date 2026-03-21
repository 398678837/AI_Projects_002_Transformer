"""
数组操作演示
Array Operations Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("数组操作 (Array Operations) 演示")
print("=" * 70)

# 1. 索引和切片
print("\n1. 索引和切片...")
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("原始数组:")
print(arr)

# 基本索引
print("\n基本索引:")
print("arr[0, 0] =", arr[0, 0])  # 第一个元素
print("arr[1, 2] =", arr[1, 2])  # 第二行第三列

# 切片
print("\n切片:")
print("arr[:, 1] =", arr[:, 1])  # 所有行的第二列
print("arr[1, :] =", arr[1, :])  # 第二行的所有列
print("arr[0:2, 1:3] =", arr[0:2, 1:3])  # 前两行，中间两列

# 布尔索引
print("\n布尔索引:")
mask = arr > 5
print("掩码:")
print(mask)
print("arr[mask] =", arr[mask])  # 大于5的元素

# 花式索引
print("\n花式索引:")
rows = [0, 1, 2]
cols = [1, 2, 3]
print("arr[rows, cols] =", arr[rows, cols])  # 按位置索引

# 2. 形状操作
print("\n2. 形状操作...")
arr = np.arange(12)
print("原始数组:", arr)
print("形状:", arr.shape)

# 重塑
reshaped = arr.reshape(3, 4)
print("\n重塑为3x4:")
print(reshaped)
print("形状:", reshaped.shape)

# 展平
flattened = reshaped.flatten()
print("\n展平:", flattened)
print("形状:", flattened.shape)

# 转置
transposed = reshaped.T
print("\n转置:")
print(transposed)
print("形状:", transposed.shape)

# 3. 数组拼接
print("\n3. 数组拼接...")
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

# 水平拼接
hstack = np.hstack((a, b))
print("\n水平拼接:")
print(hstack)

# 深度拼接
dstack = np.dstack((a, b))
print("\n深度拼接:")
print(dstack)
print("形状:", dstack.shape)

# 4. 分割
print("\n4. 分割...")
arr = np.arange(12).reshape(3, 4)
print("原始数组:")
print(arr)

# 垂直分割
vsplit = np.vsplit(arr, 3)
print("\n垂直分割:")
for i, part in enumerate(vsplit):
    print(f"部分 {i+1}:")
    print(part)

# 水平分割
hsplit = np.hsplit(arr, 2)
print("\n水平分割:")
for i, part in enumerate(hsplit):
    print(f"部分 {i+1}:")
    print(part)

# 5. 重复
print("\n5. 重复...")
arr = np.array([1, 2, 3])
print("原始数组:", arr)

# 重复元素
repeated = np.repeat(arr, 3)
print("\n重复元素:", repeated)

# 重复数组
tiled = np.tile(arr, 3)
print("\n重复数组:", tiled)

# 6. 排序
print("\n6. 排序...")
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

# 7. 唯一值
print("\n7. 唯一值...")
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5])
print("原始数组:", arr)

# 唯一值
unique = np.unique(arr)
print("\n唯一值:", unique)

# 唯一值及其计数
unique, counts = np.unique(arr, return_counts=True)
print("\n唯一值及其计数:")
print(dict(zip(unique, counts)))

# 8. 可视化
print("\n8. 可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 操作性能
ax = axes[0, 0]
operations = ['索引', '切片', '重塑', '拼接', '排序']
speed = [95, 90, 85, 80, 75]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(operations, speed, color=colors, alpha=0.7)
ax.set_xlabel('操作类型', fontsize=10)
ax.set_ylabel('相对速度(%)', fontsize=10)
ax.set_title('不同操作的性能', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, spd in zip(bars, speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{spd}%', ha='center', va='bottom', fontsize=9)

# 内存占用
ax = axes[0, 1]
operations = ['索引', '切片', '重塑', '拼接', '排序']
memory = [10, 20, 15, 40, 25]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(operations, memory, color=colors, alpha=0.7)
ax.set_xlabel('操作类型', fontsize=10)
ax.set_ylabel('内存占用(相对单位)', fontsize=10)
ax.set_title('不同操作的内存占用', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, mem in zip(bars, memory):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{mem}', ha='center', va='bottom', fontsize=9)

# 索引类型性能
ax = axes[0, 2]
index_types = ['基本索引', '切片', '布尔索引', '花式索引']
speed = [95, 90, 75, 65]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(index_types, speed, color=colors, alpha=0.7)
ax.set_xlabel('索引类型', fontsize=10)
ax.set_ylabel('相对速度(%)', fontsize=10)
ax.set_title('不同索引类型的性能', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, spd in zip(bars, speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{spd}%', ha='center', va='bottom', fontsize=9)

# 形状操作
ax = axes[1, 0]
shape_ops = ['reshape', 'flatten', 'ravel', 'transpose']
popularity = [90, 85, 80, 95]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(shape_ops, popularity, color=colors, alpha=0.7)
ax.set_xlabel('形状操作', fontsize=10)
ax.set_ylabel('使用频率(%)', fontsize=10)
ax.set_title('形状操作使用频率', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, pop in zip(bars, popularity):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{pop}%', ha='center', va='bottom', fontsize=9)

# 拼接方法
ax = axes[1, 1]
join_ops = ['vstack', 'hstack', 'dstack', 'concatenate']
usage = [40, 35, 15, 10]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(join_ops, usage, color=colors, alpha=0.7)
ax.set_xlabel('拼接方法', fontsize=10)
ax.set_ylabel('使用比例(%)', fontsize=10)
ax.set_title('拼接方法使用比例', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, use in zip(bars, usage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{use}%', ha='center', va='bottom', fontsize=9)

# 排序算法
ax = axes[1, 2]
sort_algo = ['快速排序', '归并排序', '堆排序']
performance = [90, 85, 80]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(sort_algo, performance, color=colors, alpha=0.7)
ax.set_xlabel('排序算法', fontsize=10)
ax.set_ylabel('性能(%)', fontsize=10)
ax.set_title('不同排序算法的性能', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, perf in zip(bars, performance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{perf}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'array_operations.png'))
print("可视化已保存为 'images/array_operations.png'")

# 9. 总结
print("\n" + "=" * 70)
print("数组操作总结")
print("=" * 70)

print("""
数组操作类型：
1. 索引和切片：
   - 基本索引：arr[0, 0]
   - 切片：arr[:, 1], arr[0:2, 1:3]
   - 布尔索引：arr[arr > 5]
   - 花式索引：arr[[0, 1], [1, 2]]

2. 形状操作：
   - 重塑：reshape()
   - 展平：flatten(), ravel()
   - 转置：T, transpose()

3. 数组拼接：
   - 垂直拼接：vstack()
   - 水平拼接：hstack()
   - 深度拼接：dstack()

4. 其他操作：
   - 分割：vsplit(), hsplit()
   - 重复：repeat(), tile()
   - 排序：sort()
   - 唯一值：unique()

性能考虑：
- 基本索引和切片最快
- 布尔索引和花式索引较慢
- 拼接操作内存开销较大
- 排序操作时间复杂度较高
""")

print("=" * 70)
print("数组操作演示完成！")
print("=" * 70)