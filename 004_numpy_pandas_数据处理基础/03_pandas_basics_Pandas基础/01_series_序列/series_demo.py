"""
序列 (Series) 演示
Series Demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("序列 (Series) 演示")
print("=" * 70)

# 1. 创建序列
print("\n1. 创建序列...")

# 从列表创建
print("从列表创建:")
data = [1, 2, 3, 4, 5]
s1 = pd.Series(data)
print(s1)

# 从NumPy数组创建
print("\n从NumPy数组创建:")
arr = np.array([10, 20, 30, 40, 50])
s2 = pd.Series(arr)
print(s2)

# 从字典创建
print("\n从字典创建:")
dict_data = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
s3 = pd.Series(dict_data)
print(s3)

# 指定索引
print("\n指定索引:")
data = [1, 2, 3, 4, 5]
index = ['x', 'y', 'z', 'w', 'v']
s4 = pd.Series(data, index=index)
print(s4)

# 2. 访问序列元素
print("\n2. 访问序列元素...")

# 通过索引访问
print("通过索引访问:")
print("s4['x']:", s4['x'])
print("s4[0]:", s4[0])

# 切片
print("\n切片:")
print(s4[1:4])
print(s4['y':'w'])

# 布尔索引
print("\n布尔索引:")
print(s4[s4 > 2])

# 3. 序列属性
print("\n3. 序列属性...")

print("s4.index:", s4.index)
print("s4.values:", s4.values)
print("s4.shape:", s4.shape)
print("s4.dtype:", s4.dtype)
print("s4.size:", s4.size)
print("s4.ndim:", s4.ndim)

# 4. 序列方法
print("\n4. 序列方法...")

# 统计方法
print("统计方法:")
print("s4.sum():", s4.sum())
print("s4.mean():", s4.mean())
print("s4.median():", s4.median())
print("s4.std():", s4.std())
print("s4.min():", s4.min())
print("s4.max():", s4.max())

# 排序
print("\n排序:")
print("s4.sort_values():")
print(s4.sort_values())
print("\ns4.sort_index():")
print(s4.sort_index())

# 唯一值
print("\n唯一值:")
s5 = pd.Series([1, 2, 2, 3, 3, 3])
print("s5.unique():", s5.unique())
print("s5.value_counts():")
print(s5.value_counts())

# 缺失值处理
print("\n缺失值处理:")
s6 = pd.Series([1, 2, np.nan, 4, np.nan])
print("s6:")
print(s6)
print("s6.isna():")
print(s6.isna())
print("s6.notna():")
print(s6.notna())
print("s6.dropna():")
print(s6.dropna())
print("s6.fillna(0):")
print(s6.fillna(0))

# 5. 序列运算
print("\n5. 序列运算...")

# 基本运算
print("基本运算:")
s7 = pd.Series([1, 2, 3, 4, 5])
print("s7 + 1:")
print(s7 + 1)
print("s7 * 2:")
print(s7 * 2)
print("s7 ** 2:")
print(s7 ** 2)

# 序列之间的运算
print("\n序列之间的运算:")
s8 = pd.Series([10, 20, 30, 40, 50])
print("s7 + s8:")
print(s7 + s8)
print("s7 * s8:")
print(s7 * s8)

# 索引对齐
print("\n索引对齐:")
s9 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s10 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
print("s9:")
print(s9)
print("s10:")
print(s10)
print("s9 + s10:")
print(s9 + s10)

# 6. 向量化操作
print("\n6. 向量化操作...")

# 应用函数
print("应用函数:")
s11 = pd.Series([1, 2, 3, 4, 5])
print("s11.apply(lambda x: x * 2):")
print(s11.apply(lambda x: x * 2))

# 数学函数
print("\n数学函数:")
print("np.sqrt(s11):")
print(np.sqrt(s11))
print("np.exp(s11):")
print(np.exp(s11))

# 字符串方法
print("\n字符串方法:")
s12 = pd.Series(['apple', 'banana', 'cherry', 'date'])
print("s12.str.upper():")
print(s12.str.upper())
print("s12.str.len():")
print(s12.str.len())

# 7. 时间序列
print("\n7. 时间序列...")

# 创建时间序列
print("创建时间序列:")
dates = pd.date_range('2023-01-01', periods=5)
s13 = pd.Series([1, 2, 3, 4, 5], index=dates)
print(s13)

# 时间序列索引
print("\n时间序列索引:")
print("s13['2023-01-02']:", s13['2023-01-02'])
print("s13['2023-01']:")
print(s13['2023-01'])

# 8. 性能测试
print("\n8. 性能测试...")
import time

# 测试不同大小序列的性能
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    # 创建序列
    s = pd.Series(np.random.rand(size))
    
    # 测试求和
    start = time.time()
    s.sum()
    end = time.time()
    times.append(end - start)
    print(f"{size}元素序列求和耗时: {end - start:.6f}秒")

# 9. 可视化
print("\n9. 可视化...")

# 创建可视化数据
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 序列值分布
ax = axes[0, 0]
s = pd.Series(np.random.normal(0, 1, 1000))
s.hist(ax=ax, bins=30, alpha=0.7, color='steelblue')
ax.set_title('序列值分布', fontsize=12)
ax.set_xlabel('值', fontsize=10)
ax.set_ylabel('频率', fontsize=10)
ax.grid(True, alpha=0.3)

# 序列折线图
ax = axes[0, 1]
dates = pd.date_range('2023-01-01', periods=30)
data = np.random.randn(30).cumsum()
s = pd.Series(data, index=dates)
s.plot(ax=ax, color='coral', linewidth=2)
ax.set_title('时间序列折线图', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3)

# 序列条形图
ax = axes[1, 0]
data = {'A': 10, 'B': 20, 'C': 15, 'D': 25, 'E': 30}
s = pd.Series(data)
s.plot(kind='bar', ax=ax, color='green', alpha=0.7)
ax.set_title('序列条形图', fontsize=12)
ax.set_xlabel('类别', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 性能对比
ax = axes[1, 1]
ax.plot(sizes, times, marker='o', color='purple')
ax.set_title('序列求和性能', fontsize=12)
ax.set_xlabel('序列大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'series_visualization.png'))
print("可视化已保存为 'images/series_visualization.png'")

# 10. 应用示例
print("\n10. 应用示例...")

# 10.1 数据统计
print("\n10.1 数据统计...")

# 模拟销售数据
sales = pd.Series([100, 150, 200, 120, 180, 250, 220], 
                 index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
print("销售数据:")
print(sales)
print("\n销售统计:")
print(f"总销售额: {sales.sum()}")
print(f"平均销售额: {sales.mean():.2f}")
print(f"最高销售额: {sales.max()}")
print(f"最低销售额: {sales.min()}")
print(f"销售额标准差: {sales.std():.2f}")

# 10.2 数据过滤
print("\n10.2 数据过滤...")

# 过滤销售额大于150的数据
print("销售额大于150的日期:")
print(sales[sales > 150])

# 10.3 数据转换
print("\n10.3 数据转换...")

# 计算销售额的百分比变化
print("销售额百分比变化:")
print(sales.pct_change() * 100)

# 11. 总结
print("\n" + "=" * 70)
print("序列 (Series) 总结")
print("=" * 70)

print("""
Pandas Series 功能：

1. 创建序列：
   - 从列表、NumPy数组、字典创建
   - 指定索引

2. 访问元素：
   - 通过索引访问
   - 切片
   - 布尔索引

3. 属性和方法：
   - 统计方法：sum, mean, median, std, min, max
   - 排序：sort_values, sort_index
   - 唯一值：unique, value_counts
   - 缺失值处理：isna, notna, dropna, fillna

4. 运算：
   - 基本运算
   - 序列之间的运算
   - 索引对齐

5. 向量化操作：
   - apply函数
   - 数学函数
   - 字符串方法

6. 时间序列：
   - 日期范围创建
   - 时间索引

7. 应用场景：
   - 数据统计
   - 数据过滤
   - 数据转换
   - 时间序列分析
""")

print("=" * 70)
print("序列 (Series) 演示完成！")
print("=" * 70)