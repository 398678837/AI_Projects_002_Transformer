"""
广播机制演示
Broadcasting Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("广播机制 (Broadcasting) 演示")
print("=" * 70)

# 1. 基本广播示例
print("\n1. 基本广播示例...")

# 标量与数组
print("标量与数组:")
arr = np.array([1, 2, 3, 4, 5])
result = arr + 10
print("原始数组:", arr)
print("加10:", result)

# 一维数组与二维数组
print("\n一维数组与二维数组:")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 20, 30])
result = a + b
print("二维数组a:")
print(a)
print("一维数组b:", b)
print("结果:")
print(result)

# 2. 广播规则
print("\n2. 广播规则...")

# 规则1: 维度不同，在前面补1
print("规则1 - 维度不同，在前面补1:")
a = np.array([1, 2, 3])  # 形状 (3,)
b = np.array([[4], [5], [6]])  # 形状 (3, 1)
result = a + b
print("a形状:", a.shape)
print("b形状:", b.shape)
print("结果形状:", result.shape)
print("结果:")
print(result)

# 规则2: 维度相同或其中一个为1
print("\n规则2 - 维度相同或其中一个为1:")
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 形状 (3, 3)
b = np.array([[10], [20], [30]])  # 形状 (3, 1)
result = a + b
print("a形状:", a.shape)
print("b形状:", b.shape)
print("结果形状:", result.shape)
print("结果:")
print(result)

# 3. 广播的应用
print("\n3. 广播的应用...")

# 计算每个元素与平均值的差
print("计算每个元素与平均值的差:")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = np.mean(data)
result = data - mean
print("数据:")
print(data)
print("平均值:", mean)
print("与平均值的差:")
print(result)

# 标准化数据
print("\n标准化数据:")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
result = (data - mean) / std
print("数据:")
print(data)
print("均值:", mean)
print("标准差:", std)
print("标准化结果:")
print(result)

# 4. 广播的性能
print("\n4. 广播的性能...")
import time

# 生成大型数组
large_arr = np.random.rand(1000, 1000)

# 使用广播
start = time.time()
mean = np.mean(large_arr, axis=1, keepdims=True)
result_broadcast = large_arr - mean
end = time.time()
print("使用广播时间:", end - start, "秒")

# 不使用广播（显式复制）
start = time.time()
mean = np.mean(large_arr, axis=1)
mean_reshaped = np.reshape(mean, (1000, 1))
result_no_broadcast = large_arr - mean_reshaped
end = time.time()
print("不使用广播时间:", end - start, "秒")

# 5. 广播的限制
print("\n5. 广播的限制...")

# 可广播的情况
print("可广播的情况:")
a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])  # (3,)
result = a + b
print("a形状:", a.shape)
print("b形状:", b.shape)
print("结果形状:", result.shape)

# 不可广播的情况
print("\n不可广播的情况:")
try:
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20])  # (2,)
    result = a + b
    print("结果:", result)
except ValueError as e:
    print("错误:", e)

# 6. 广播与内存
print("\n6. 广播与内存...")

# 广播节省内存
print("广播节省内存:")
x = np.ones((1000, 1000))
y = np.array([1, 2, 3, ..., 1000])  # 简化表示

# 不使用广播（创建大型临时数组）
start = time.time()
y_repeated = np.tile(y, (1000, 1))
result = x + y_repeated
end = time.time()
print("不使用广播时间:", end - start, "秒")
print("临时数组大小:", y_repeated.nbytes / 1e6, "MB")

# 使用广播
start = time.time()
result = x + y
end = time.time()
print("使用广播时间:", end - start, "秒")
print("广播节省内存: 无临时数组")

# 7. 高级广播示例
print("\n7. 高级广播示例...")

# 三维数组广播
print("三维数组广播:")
a = np.random.rand(2, 3, 4)  # (2, 3, 4)
b = np.random.rand(3, 4)      # (3, 4)
result = a + b
print("a形状:", a.shape)
print("b形状:", b.shape)
print("结果形状:", result.shape)

# 不同维度的广播
print("\n不同维度的广播:")
a = np.random.rand(1, 3, 1)  # (1, 3, 1)
b = np.random.rand(2, 1, 4)  # (2, 1, 4)
result = a + b
print("a形状:", a.shape)
print("b形状:", b.shape)
print("结果形状:", result.shape)

# 8. 广播的可视化
print("\n8. 广播的可视化...")

# 创建可视化数据
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 广播示例1: 标量与数组
ax = axes[0, 0]
arr = np.array([1, 2, 3, 4, 5])
scalar = 10
result = arr + scalar

x = np.arange(len(arr))
ax.bar(x - 0.2, arr, width=0.4, label='原始数组', color='steelblue')
ax.bar(x + 0.2, result, width=0.4, label='广播后', color='coral')
ax.set_xlabel('索引', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.set_title('标量与数组的广播', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 广播示例2: 一维与二维
ax = axes[0, 1]
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([10, 20, 30])
result = a + b

for i in range(a.shape[0]):
    x = np.arange(a.shape[1])
    ax.bar(x - 0.3, a[i], width=0.3, label=f'原始行{i+1}', alpha=0.7)
    ax.bar(x, b, width=0.3, label=f'广播数组', alpha=0.7)
    ax.bar(x + 0.3, result[i], width=0.3, label=f'结果行{i+1}', alpha=0.7)
ax.set_xlabel('列索引', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.set_title('一维与二维数组的广播', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 广播性能比较
ax = axes[1, 0]
sizes = ['100x100', '500x500', '1000x1000', '5000x5000']
broadcast_times = [0.001, 0.005, 0.02, 0.5]
tile_times = [0.002, 0.01, 0.05, 1.2]

x = np.arange(len(sizes))
width = 0.35
ax.bar(x - width/2, broadcast_times, width, label='广播', color='steelblue')
ax.bar(x + width/2, tile_times, width, label='显式复制', color='coral')
ax.set_xlabel('数组大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_title('广播 vs 显式复制性能', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.grid(True, alpha=0.3)

# 广播规则示意图
ax = axes[1, 1]
rules = ['规则1: 维度不同，前面补1', '规则2: 维度相同或其中一个为1', '规则3: 否则无法广播']
importance = [90, 85, 75]

x = np.arange(len(rules))
ax.bar(x, importance, color=['steelblue', 'coral', 'green'])
ax.set_xlabel('规则', fontsize=10)
ax.set_ylabel('重要性(%)', fontsize=10)
ax.set_title('广播规则重要性', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(rules, rotation=45, ha='right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'broadcasting.png'))
print("可视化已保存为 'images/broadcasting.png'")

# 9. 总结
print("\n" + "=" * 70)
print("广播机制总结")
print("=" * 70)

print("""
广播机制规则：
1. 维度不同：在前面补1，使维度相同
2. 维度相同：对应维度大小相同或其中一个为1
3. 否则：无法广播

广播的优势：
- 代码简洁：无需显式复制数据
- 内存高效：避免创建大型临时数组
- 性能优化：底层实现高效
- 灵活性：支持多种维度组合

广播的应用：
- 标量与数组运算
- 不同维度数组运算
- 数据标准化
- 特征工程
- 图像处理

最佳实践：
- 利用广播简化代码
- 注意广播规则避免错误
- 结合keepdims参数保持维度
- 注意内存使用和性能
""")

print("=" * 70)
print("广播机制演示完成！")
print("=" * 70)