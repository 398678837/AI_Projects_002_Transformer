"""
数据框 (DataFrame) 演示
DataFrame Demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("数据框 (DataFrame) 演示")
print("=" * 70)

# 1. 创建数据框
print("\n1. 创建数据框...")

# 从字典创建
print("从字典创建:")
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 40, 45],
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
    'salary': [50000, 60000, 70000, 80000, 90000]
}
df1 = pd.DataFrame(data)
print(df1)

# 从NumPy数组创建
print("\n从NumPy数组创建:")
arr = np.random.rand(5, 3)
df2 = pd.DataFrame(arr, columns=['A', 'B', 'C'], index=['X', 'Y', 'Z', 'W', 'V'])
print(df2)

# 从列表创建
print("\n从列表创建:")
data_list = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'London'],
    ['Charlie', 35, 'Paris']
]
df3 = pd.DataFrame(data_list, columns=['name', 'age', 'city'])
print(df3)

# 2. 查看数据框
print("\n2. 查看数据框...")

print("df1.head():")
print(df1.head())
print("\ndf1.tail():")
print(df1.tail())
print("\ndf1.shape:", df1.shape)
print("df1.columns:", df1.columns)
print("df1.index:", df1.index)
print("\ndf1.info():")
df1.info()
print("\ndf1.describe():")
print(df1.describe())

# 3. 访问数据
print("\n3. 访问数据...")

# 访问列
print("访问列:")
print("df1['name']:")
print(df1['name'])
print("\ndf1.age:")
print(df1.age)

# 访问多列
print("\n访问多列:")
print(df1[['name', 'salary']])

# 访问行
print("\n访问行:")
print("df1.loc[0]:")
print(df1.loc[0])
print("\ndf1.iloc[0]:")
print(df1.iloc[0])

# 访问多行
print("\n访问多行:")
print(df1.loc[1:3])
print("\ndf1.iloc[1:3]")
print(df1.iloc[1:3])

# 访问特定单元格
print("\n访问特定单元格:")
print("df1.loc[0, 'name']:", df1.loc[0, 'name'])
print("df1.iloc[0, 0]:", df1.iloc[0, 0])

# 4. 数据操作
print("\n4. 数据操作...")

# 添加列
print("添加列:")
df1['bonus'] = df1['salary'] * 0.1
print(df1)

# 删除列
print("\n删除列:")
df1 = df1.drop('bonus', axis=1)
print(df1)

# 添加行
print("\n添加行:")
new_row = {'name': 'Frank', 'age': 50, 'city': 'Berlin', 'salary': 100000}
df1 = df1.append(new_row, ignore_index=True)
print(df1)

# 删除行
print("\n删除行:")
df1 = df1.drop(5, axis=0)
print(df1)

# 5. 数据过滤
print("\n5. 数据过滤...")

# 布尔索引
print("布尔索引:")
print(df1[df1['age'] > 30])

# 多条件过滤
print("\n多条件过滤:")
print(df1[(df1['age'] > 30) & (df1['salary'] > 60000)])

# 字符串过滤
print("\n字符串过滤:")
print(df1[df1['city'].str.contains('N')])

# 6. 数据排序
print("\n6. 数据排序...")

# 按列排序
print("按salary列排序:")
print(df1.sort_values('salary', ascending=False))

# 按多列排序
print("\n按age和salary排序:")
print(df1.sort_values(['age', 'salary'], ascending=[True, False]))

# 7. 数据分组
print("\n7. 数据分组...")

# 按city分组
print("按city分组:")
grouped = df1.groupby('city')
print(grouped.size())

# 分组聚合
print("\n分组聚合:")
print(grouped['salary'].mean())

# 多聚合函数
print("\n多聚合函数:")
print(grouped['salary'].agg(['mean', 'sum', 'count']))

# 8. 缺失值处理
print("\n8. 缺失值处理...")

# 创建包含缺失值的数据框
df_missing = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, np.nan, 35, 40],
    'salary': [50000, 60000, np.nan, 80000]
})
print("原始数据:")
print(df_missing)

# 检测缺失值
print("\n检测缺失值:")
print(df_missing.isna())

# 填充缺失值
print("\n填充缺失值:")
print(df_missing.fillna(0))
print("\n填充缺失值（均值）:")
print(df_missing.fillna(df_missing.mean()))

# 丢弃缺失值
print("\n丢弃缺失值:")
print(df_missing.dropna())

# 9. 数据合并
print("\n9. 数据合并...")

# 创建两个数据框
df_left = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40]
})

df_right = pd.DataFrame({
    'id': [1, 2, 3, 5],
    'salary': [50000, 60000, 70000, 80000],
    'city': ['New York', 'London', 'Paris', 'Tokyo']
})

print("左数据框:")
print(df_left)
print("\n右数据框:")
print(df_right)

# 内连接
print("\n内连接:")
print(pd.merge(df_left, df_right, on='id', how='inner'))

# 左连接
print("\n左连接:")
print(pd.merge(df_left, df_right, on='id', how='left'))

# 右连接
print("\n右连接:")
print(pd.merge(df_left, df_right, on='id', how='right'))

# 外连接
print("\n外连接:")
print(pd.merge(df_left, df_right, on='id', how='outer'))

# 10. 数据透视表
print("\n10. 数据透视表...")

# 创建示例数据
df_pivot = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=12, freq='M'),
    'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'value': np.random.randint(100, 1000, 12)
})
print("原始数据:")
print(df_pivot)

# 创建数据透视表
print("\n数据透视表:")
pivot_table = pd.pivot_table(df_pivot, values='value', index='date', columns='category', aggfunc=np.sum)
print(pivot_table)

# 11. 性能测试
print("\n11. 性能测试...")
import time

# 测试不同大小数据框的性能
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    # 创建数据框
    df = pd.DataFrame({
        'A': np.random.rand(size),
        'B': np.random.rand(size),
        'C': np.random.rand(size),
        'D': np.random.rand(size)
    })
    
    # 测试求和
    start = time.time()
    df.sum()
    end = time.time()
    times.append(end - start)
    print(f"{size}行数据框求和耗时: {end - start:.6f}秒")

# 12. 可视化
print("\n12. 可视化...")

# 创建可视化数据
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 柱状图
ax = axes[0, 0]
df1.plot(kind='bar', x='name', y='salary', ax=ax, color='steelblue')
ax.set_title('薪资分布', fontsize=12)
ax.set_xlabel('姓名', fontsize=10)
ax.set_ylabel('薪资', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 折线图
ax = axes[0, 1]
dates = pd.date_range('2023-01-01', periods=12, freq='M')
values = np.random.randn(12).cumsum()
df_time = pd.DataFrame({'date': dates, 'value': values})
df_time.plot(x='date', y='value', ax=ax, color='coral', linewidth=2)
ax.set_title('时间序列', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3)

# 散点图
ax = axes[1, 0]
df_scatter = pd.DataFrame({
    'x': np.random.rand(100),
    'y': np.random.rand(100)
})
df_scatter.plot(kind='scatter', x='x', y='y', ax=ax, color='green', alpha=0.6)
ax.set_title('散点图', fontsize=12)
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.grid(True, alpha=0.3)

# 性能对比
ax = axes[1, 1]
ax.plot(sizes, times, marker='o', color='purple')
ax.set_title('数据框求和性能', fontsize=12)
ax.set_xlabel('数据框大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'dataframe_visualization.png'))
print("可视化已保存为 'images/dataframe_visualization.png'")

# 13. 应用示例
print("\n13. 应用示例...")

# 13.1 数据分析
print("\n13.1 数据分析...")

# 模拟销售数据
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365, freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], 365),
    'sales': np.random.randint(100, 1000, 365),
    'profit': np.random.randint(10, 100, 365)
})

# 按产品分组分析
print("按产品分组分析:")
product_analysis = sales_data.groupby('product').agg({
    'sales': ['sum', 'mean', 'std'],
    'profit': ['sum', 'mean', 'std']
})
print(product_analysis)

# 按月分析
print("\n按月分析:")
sales_data['month'] = sales_data['date'].dt.month
monthly_analysis = sales_data.groupby('month').agg({
    'sales': 'sum',
    'profit': 'sum'
})
print(monthly_analysis)

# 13.2 数据清洗
print("\n13.2 数据清洗...")

# 创建包含脏数据的示例
 dirty_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, '30', 35, np.nan, 45],
    'salary': [50000, 60000, '70000', 80000, 90000]
})

print("原始数据:")
print(dirty_data)
print("\n数据类型:")
print(dirty_data.dtypes)

# 转换数据类型
dirty_data['age'] = pd.to_numeric(dirty_data['age'], errors='coerce')
dirty_data['salary'] = pd.to_numeric(dirty_data['salary'], errors='coerce')

# 填充缺失值
dirty_data['age'] = dirty_data['age'].fillna(dirty_data['age'].mean())

print("\n清洗后数据:")
print(dirty_data)
print("\n数据类型:")
print(dirty_data.dtypes)

# 14. 总结
print("\n" + "=" * 70)
print("数据框 (DataFrame) 总结")
print("=" * 70)

print("""
Pandas DataFrame 功能：

1. 创建数据框：
   - 从字典创建
   - 从NumPy数组创建
   - 从列表创建

2. 查看数据：
   - head(), tail()
   - shape, columns, index
   - info(), describe()

3. 访问数据：
   - 访问列：df['column'] 或 df.column
   - 访问行：loc[], iloc[]
   - 访问单元格：df.loc[row, column]

4. 数据操作：
   - 添加/删除列
   - 添加/删除行
   - 数据过滤
   - 数据排序

5. 数据处理：
   - 缺失值处理：fillna(), dropna()
   - 数据分组：groupby()
   - 数据合并：merge()
   - 数据透视表：pivot_table()

6. 应用场景：
   - 数据分析
   - 数据清洗
   - 数据可视化
   - 机器学习数据准备

7. 性能考虑：
   - 大型数据框的处理
   - 内存使用优化
   - 计算效率
""")

print("=" * 70)
print("数据框 (DataFrame) 演示完成！")
print("=" * 70)