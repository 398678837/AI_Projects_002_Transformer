# 分组 (GroupBy) 详细文档

## 1. 什么是分组

分组（GroupBy）是Pandas库中的一个强大功能，用于将数据按照一个或多个键进行分组，然后对每个组应用聚合函数。分组操作是数据分析中的核心操作之一，类似于SQL中的GROUP BY语句。

### 1.1 分组的基本原理

分组操作通常分为三个步骤：

1. **拆分（Split）**：将数据按照指定的键拆分成多个组
2. **应用（Apply）**：对每个组应用聚合函数
3. **合并（Combine）**：将结果合并成一个新的数据结构

### 1.2 分组的应用场景

- **数据分析**：按类别分析数据的统计特征
- **业务报表**：生成按部门、地区等维度的汇总报表
- **数据预处理**：按组进行数据转换和清洗
- **特征工程**：按组计算特征值

## 2. 基本分组

### 2.1 按单列分组

```python
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'sales': [100, 200, 150, 250, 120, 220],
    'profit': [10, 20, 15, 25, 12, 22]
})

# 按产品分组
grouped = data.groupby('product')

# 查看分组大小
print(grouped.size())

# 查看每个分组
for name, group in grouped:
    print(f"Group: {name}")
    print(group)
    print()
```

### 2.2 按多列分组

```python
# 按多列分组
data['category'] = ['Electronics', 'Clothing', 'Electronics', 'Clothing', 'Electronics', 'Clothing']
grouped = data.groupby(['product', 'category'])

# 查看分组大小
print(grouped.size())

# 查看每个分组
for name, group in grouped:
    print(f"Group: {name}")
    print(group)
    print()
```

### 2.3 按函数分组

```python
# 按函数分组
data['date'] = pd.date_range('2023-01-01', periods=6, freq='M')

def get_quarter(date):
    return date.quarter

grouped = data.groupby(get_quarter, level='date')
print(grouped.size())
```

## 3. 分组聚合

### 3.1 内置聚合函数

```python
# 内置聚合函数
print("按产品聚合销售数据:")
print(grouped['sales'].sum())

print("\n按产品聚合利润数据:")
print(grouped['profit'].mean())

print("\n按产品聚合多个列:")
print(grouped[['sales', 'profit']].sum())
```

### 3.2 多函数聚合

```python
# 多函数聚合
print("按产品计算销售的统计指标:")
print(grouped['sales'].agg(['sum', 'mean', 'std', 'min', 'max']))

# 为聚合结果命名
print("\n为聚合结果命名:")
print(grouped['sales'].agg([('总销售额', 'sum'), ('平均销售额', 'mean'), ('销售标准差', 'std')]))
```

### 3.3 自定义聚合函数

```python
# 自定义聚合函数
def range_func(x):
    return x.max() - x.min()

print("按产品计算销售范围:")
print(grouped['sales'].agg(range_func))

# 多种聚合函数
print("\n按产品计算多种统计指标:")
print(grouped['sales'].agg([('总销售额', 'sum'), ('平均销售额', 'mean'), ('销售范围', range_func)]))
```

## 4. 分组操作

### 4.1 apply 方法

`apply` 方法用于对每个分组应用任意函数。

```python
# apply 方法
print("分组后应用函数:")
print(grouped.apply(lambda x: x['sales'].sum()))

# 应用复杂函数
def analyze_group(group):
    return pd.Series({
        'total_sales': group['sales'].sum(),
        'average_profit': group['profit'].mean(),
        'sales_range': group['sales'].max() - group['sales'].min()
    })

print("\n应用复杂函数:")
print(grouped.apply(analyze_group))
```

### 4.2 transform 方法

`transform` 方法用于对每个分组应用函数，并返回与原数据形状相同的结果。

```python
# transform 方法
print("分组后转换:")
print(grouped['sales'].transform(lambda x: x / x.sum()))

# 标准化数据
print("\n标准化数据:")
print(grouped['sales'].transform(lambda x: (x - x.mean()) / x.std()))
```

### 4.3 filter 方法

`filter` 方法用于根据条件过滤分组。

```python
# filter 方法
print("分组后过滤:")
print(grouped.filter(lambda x: x['sales'].sum() > 300))

# 过滤平均销售额大于150的组
print("\n过滤平均销售额大于150的组:")
print(grouped.filter(lambda x: x['sales'].mean() > 150))
```

## 5. 分组中的时间序列

### 5.1 按时间周期分组

```python
# 按时间周期分组
data['date'] = pd.date_range('2023-01-01', periods=365, freq='D')
data['sales'] = np.random.randint(100, 1000, 365)

# 按月分组
data['month'] = data['date'].dt.month
grouped_month = data.groupby('month')
print("按月分组销售数据:")
print(grouped_month['sales'].sum())

# 按季度分组
data['quarter'] = data['date'].dt.quarter
grouped_quarter = data.groupby('quarter')
print("\n按季度分组销售数据:")
print(grouped_quarter['sales'].sum())
```

### 5.2 时间序列重采样

```python
# 时间序列重采样
data.set_index('date', inplace=True)

# 按月重采样
monthly = data.resample('M').sum()
print("按月重采样:")
print(monthly)

# 按季度重采样
quarterly = data.resample('Q').sum()
print("\n按季度重采样:")
print(quarterly)
```

## 6. 分组透视

### 6.1 基本透视表

```python
# 基本透视表
pivot_table = data.pivot_table(values='sales', index='product', columns='category', aggfunc=np.sum)
print("产品和类别销售透视表:")
print(pivot_table)
```

### 6.2 多值透视表

```python
# 多值透视表
pivot_table_multi = data.pivot_table(values=['sales', 'profit'], index='product', columns='category', aggfunc=np.sum)
print("产品和类别销售与利润透视表:")
print(pivot_table_multi)
```

### 6.3 多级透视表

```python
# 多级透视表
pivot_table_multi_level = data.pivot_table(values='sales', index=['product', 'category'], columns='region', aggfunc=np.sum)
print("多级透视表:")
print(pivot_table_multi_level)
```

## 7. 性能优化

### 7.1 大型数据的分组性能

```python
# 测试大型数据的分组性能
import time

size = 1000000
data = pd.DataFrame({
    'group': np.random.choice(['A', 'B', 'C', 'D'], size),
    'value1': np.random.randn(size),
    'value2': np.random.randn(size)
})

start = time.time()
grouped = data.groupby('group').agg({'value1': 'sum', 'value2': 'mean'})
end = time.time()
print(f"{size}行数据分组聚合耗时: {end - start:.6f}秒")
```

### 7.2 内存优化

```python
# 内存优化
# 1. 使用分类数据类型
data['group'] = data['group'].astype('category')

# 2. 减少数据类型大小
data['value1'] = data['value1'].astype('float32')
data['value2'] = data['value2'].astype('float32')

# 3. 避免不必要的列
data = data[['group', 'value1', 'value2']]
```

### 7.3 计算优化

```python
# 计算优化
# 1. 选择合适的聚合函数
# 2. 避免使用apply，尽量使用内置聚合函数
# 3. 对于复杂计算，考虑使用NumPy向量化操作
```

## 8. 应用示例

### 8.1 销售分析

```python
# 销售分析
# 按产品和月份分析销售趋势
product_monthly = data.groupby(['product', 'month'])['sales'].sum().unstack()
print("产品月度销售趋势:")
print(product_monthly)

# 分析销售增长率
grouped = data.groupby('product')
sales_growth = grouped['sales'].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
print("\n产品销售增长率:")
print(sales_growth)
```

### 8.2 客户分析

```python
# 客户分析
# 按客户分组分析购买行为
customer_analysis = data.groupby('customer_id').agg({
    'order_count': 'count',
    'total_spend': 'sum',
    'average_spend': 'mean'
})
print("客户分析:")
print(customer_analysis)
```

### 8.3 财务分析

```python
# 财务分析
# 按部门分析成本和收入
department_analysis = data.groupby('department').agg({
    'revenue': 'sum',
    'cost': 'sum'
})
department_analysis['profit'] = department_analysis['revenue'] - department_analysis['cost']
department_analysis['profit_margin'] = department_analysis['profit'] / department_analysis['revenue'] * 100
print("部门财务分析:")
print(department_analysis)
```

## 9. 常见问题和解决方案

### 9.1 分组键包含缺失值

**问题**：分组键包含缺失值时，这些行会被排除在分组之外。

**解决方案**：
- 可以使用`dropna=False`参数保留缺失值分组
- 或者在分组前处理缺失值

```python
# 保留缺失值分组
grouped = data.groupby('column_with_na', dropna=False)
```

### 9.2 分组后索引问题

**问题**：分组后结果的索引是分组键，可能不是连续的。

**解决方案**：
- 使用`reset_index()`重置索引
- 或者在分组时指定`as_index=False`

```python
# 不使用分组键作为索引
grouped = data.groupby('product', as_index=False).sum()
```

### 9.3 性能问题

**问题**：处理大型数据时分组操作速度慢。

**解决方案**：
- 优化数据类型
- 使用`categorical`类型存储分组键
- 减少不必要的列
- 考虑使用Dask等分布式计算库

## 10. 总结

分组（GroupBy）是Pandas库中一个强大的功能，用于将数据按照指定的键进行分组，然后对每个组应用聚合函数。它在数据分析、业务报表、数据预处理等场景中有着广泛的应用。

### 10.1 核心功能

- **基本分组**：按单列、多列或函数分组
- **分组聚合**：使用内置或自定义聚合函数
- **分组操作**：apply、transform、filter方法
- **时间序列分组**：按时间周期分组和重采样
- **分组透视**：创建数据透视表

### 10.2 应用场景

- **销售分析**：按产品、地区、时间分析销售数据
- **客户分析**：分析客户购买行为和价值
- **财务分析**：按部门、项目分析成本和收入
- **数据预处理**：按组进行数据清洗和转换
- **特征工程**：按组计算特征值

### 10.3 最佳实践

- **选择合适的分组键**：根据分析需求选择合适的分组键
- **使用内置聚合函数**：内置函数比自定义函数更快
- **优化数据类型**：使用合适的数据类型减少内存使用
- **避免链式操作**：减少不必要的中间计算
- **使用透视表**：对于多维度分析，使用透视表更直观

### 10.4 进阶学习

- 高级分组技巧
- 与其他Pandas功能的结合使用
- 性能优化策略
- 分布式计算库的使用