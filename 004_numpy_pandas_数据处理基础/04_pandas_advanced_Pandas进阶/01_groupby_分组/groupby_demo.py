"""
分组 (GroupBy) 演示
GroupBy Demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("分组 (GroupBy) 演示")
print("=" * 70)

# 1. 创建示例数据
print("\n1. 创建示例数据...")

# 销售数据
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365, freq='D'),
    'product': np.random.choice(['A', 'B', 'C', 'D'], 365),
    'category': np.random.choice(['Electronics', 'Clothing', 'Furniture', 'Food'], 365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
    'sales': np.random.randint(100, 1000, 365),
    'profit': np.random.randint(10, 100, 365)
})

print("示例数据:")
print(sales_data.head())
print("\n数据形状:", sales_data.shape)

# 2. 基本分组
print("\n2. 基本分组...")

# 按产品分组
print("按产品分组:")
grouped_product = sales_data.groupby('product')
print(grouped_product.size())

# 按类别分组
print("\n按类别分组:")
grouped_category = sales_data.groupby('category')
print(grouped_category.size())

# 按区域分组
print("\n按区域分组:")
grouped_region = sales_data.groupby('region')
print(grouped_region.size())

# 3. 分组聚合
print("\n3. 分组聚合...")

# 按产品聚合销售数据
print("按产品聚合销售数据:")
print(grouped_product['sales'].sum())

# 按类别聚合利润数据
print("\n按类别聚合利润数据:")
print(grouped_category['profit'].sum())

# 多列聚合
print("\n按产品聚合销售和利润:")
print(grouped_product[['sales', 'profit']].sum())

# 4. 多函数聚合
print("\n4. 多函数聚合...")

# 按产品计算销售的统计指标
print("按产品计算销售的统计指标:")
print(grouped_product['sales'].agg(['sum', 'mean', 'std', 'min', 'max']))

# 按类别计算利润的统计指标
print("\n按类别计算利润的统计指标:")
print(grouped_category['profit'].agg(['sum', 'mean', 'std', 'min', 'max']))

# 5. 自定义聚合函数
print("\n5. 自定义聚合函数...")

# 自定义聚合函数
def range_func(x):
    return x.max() - x.min()

print("按产品计算销售范围:")
print(grouped_product['sales'].agg(range_func))

# 多种聚合函数
print("\n按产品计算多种统计指标:")
print(grouped_product['sales'].agg([('总销售额', 'sum'), ('平均销售额', 'mean'), ('销售范围', range_func)]))

# 6. 多列分组
print("\n6. 多列分组...")

# 按产品和类别分组
print("按产品和类别分组:")
grouped_product_category = sales_data.groupby(['product', 'category'])
print(grouped_product_category.size())

# 按产品和类别聚合销售数据
print("\n按产品和类别聚合销售数据:")
print(grouped_product_category['sales'].sum())

# 按区域和产品分组
print("\n按区域和产品分组:")
grouped_region_product = sales_data.groupby(['region', 'product'])
print(grouped_region_product['profit'].sum())

# 7. 分组后的数据操作
print("\n7. 分组后的数据操作...")

# 分组后应用函数
print("分组后应用函数:")
print(grouped_product.apply(lambda x: x['sales'].sum()))

# 分组后转换
print("\n分组后转换:")
print(grouped_product['sales'].transform(lambda x: x / x.sum()))

# 分组后过滤
print("\n分组后过滤:")
print(grouped_product.filter(lambda x: x['sales'].sum() > 10000))

# 8. 分组中的时间序列
print("\n8. 分组中的时间序列...")

# 按月份分组
sales_data['month'] = sales_data['date'].dt.month
grouped_month = sales_data.groupby('month')
print("按月分组销售数据:")
print(grouped_month['sales'].sum())

# 按季度分组
sales_data['quarter'] = sales_data['date'].dt.quarter
grouped_quarter = sales_data.groupby('quarter')
print("\n按季度分组销售数据:")
print(grouped_quarter['sales'].sum())

# 9. 分组透视
print("\n9. 分组透视...")

# 按产品和区域透视
pivot_table = sales_data.pivot_table(values='sales', index='product', columns='region', aggfunc=np.sum)
print("产品和区域销售透视表:")
print(pivot_table)

# 多值透视
pivot_table_multi = sales_data.pivot_table(values=['sales', 'profit'], index='product', columns='category', aggfunc=np.sum)
print("\n产品和类别销售与利润透视表:")
print(pivot_table_multi)

# 10. 性能测试
print("\n10. 性能测试...")
import time

# 测试不同大小数据的分组性能
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    # 创建测试数据
    test_data = pd.DataFrame({
        'group': np.random.choice(['A', 'B', 'C', 'D'], size),
        'value1': np.random.randn(size),
        'value2': np.random.randn(size)
    })
    
    # 测试分组聚合性能
    start = time.time()
    test_data.groupby('group').agg({'value1': 'sum', 'value2': 'mean'})
    end = time.time()
    times.append(end - start)
    print(f"{size}行数据分组聚合耗时: {end - start:.6f}秒")

# 11. 可视化
print("\n11. 可视化...")

# 创建可视化数据
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 按产品分组的销售数据
ax = axes[0, 0]
product_sales = grouped_product['sales'].sum()
product_sales.plot(kind='bar', ax=ax, color='steelblue')
ax.set_title('各产品销售总额', fontsize=12)
ax.set_xlabel('产品', fontsize=10)
ax.set_ylabel('销售额', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 按类别分组的利润数据
ax = axes[0, 1]
category_profit = grouped_category['profit'].sum()
category_profit.plot(kind='bar', ax=ax, color='coral')
ax.set_title('各类别利润总额', fontsize=12)
ax.set_xlabel('类别', fontsize=10)
ax.set_ylabel('利润', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# 按区域分组的销售数据
ax = axes[1, 0]
region_sales = grouped_region['sales'].sum()
region_sales.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
ax.set_title('各区域销售占比', fontsize=12)
ax.set_ylabel('')

# 性能对比
ax = axes[1, 1]
ax.plot(sizes, times, marker='o', color='purple')
ax.set_title('分组聚合性能', fontsize=12)
ax.set_xlabel('数据大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'groupby_visualization.png'))
print("可视化已保存为 'images/groupby_visualization.png'")

# 12. 应用示例
print("\n12. 应用示例...")

# 12.1 销售分析
print("\n12.1 销售分析...")

# 按产品和月份分析销售趋势
product_monthly = sales_data.groupby(['product', 'month'])['sales'].sum().unstack()
print("产品月度销售趋势:")
print(product_monthly)

# 12.2 利润分析
print("\n12.2 利润分析...")

# 计算各产品的利润率
sales_data['profit_margin'] = sales_data['profit'] / sales_data['sales']
product_margin = sales_data.groupby('product')['profit_margin'].mean()
print("各产品利润率:")
print(product_margin)

# 12.3 区域分析
print("\n12.3 区域分析...")

# 分析各区域的销售和利润
region_analysis = sales_data.groupby('region').agg({
    'sales': 'sum',
    'profit': 'sum',
    'profit_margin': 'mean'
})
print("各区域分析:")
print(region_analysis)

# 13. 总结
print("\n" + "=" * 70)
print("分组 (GroupBy) 总结")
print("=" * 70)

print("""
Pandas GroupBy 功能：

1. 基本分组：
   - 按单列分组
   - 按多列分组
   - 按时间序列分组

2. 分组聚合：
   - 内置聚合函数：sum, mean, std, min, max, count
   - 自定义聚合函数
   - 多函数聚合

3. 分组操作：
   - apply：应用函数到每个分组
   - transform：转换每个分组的数据
   - filter：根据条件过滤分组

4. 分组透视：
   - pivot_table：创建数据透视表
   - 多维度透视

5. 应用场景：
   - 销售分析
   - 利润分析
   - 客户分析
   - 时间序列分析

6. 性能考虑：
   - 大型数据的分组性能
   - 内存使用
   - 计算效率
""")

print("=" * 70)
print("分组 (GroupBy) 演示完成！")
print("=" * 70)