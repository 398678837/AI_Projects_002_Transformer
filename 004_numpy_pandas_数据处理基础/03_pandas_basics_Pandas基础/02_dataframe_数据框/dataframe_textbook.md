# 数据框 (DataFrame) 教材

## 第一章：数据框基础

### 1.1 什么是数据框

数据框（DataFrame）是Pandas库中的一种二维数据结构，类似于电子表格或SQL表。它由行和列组成，每列可以是不同的数据类型。数据框是Pandas中最常用的数据结构，用于数据分析和处理。

### 1.2 数据框的特点

- **二维结构**：由行和列组成
- **异构数据**：每列可以是不同的数据类型
- **标签索引**：行和列都有标签
- **向量化操作**：支持向量化运算
- **缺失值处理**：内置缺失值处理功能
- **与NumPy集成**：可以与NumPy数组无缝转换

### 1.3 数据框的应用场景

- **数据分析**：统计分析、数据挖掘
- **数据清洗**：处理缺失值、异常值
- **数据可视化**：生成图表
- **机器学习**：数据准备、特征工程
- **金融分析**：时间序列分析、风险评估
- **科学研究**：实验数据处理

## 第二章：创建数据框

### 2.1 从字典创建

```python
import pandas as pd

# 从字典创建
 data = {
     'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
     'age': [25, 30, 35, 40, 45],
     'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
     'salary': [50000, 60000, 70000, 80000, 90000]
 }
 df = pd.DataFrame(data)
 print(df)
```

### 2.2 从NumPy数组创建

```python
import numpy as np

# 从NumPy数组创建
 arr = np.random.rand(5, 3)
 df = pd.DataFrame(arr, columns=['A', 'B', 'C'], index=['X', 'Y', 'Z', 'W', 'V'])
 print(df)
```

### 2.3 从列表创建

```python
# 从列表创建
 data_list = [
     ['Alice', 25, 'New York'],
     ['Bob', 30, 'London'],
     ['Charlie', 35, 'Paris']
 ]
 df = pd.DataFrame(data_list, columns=['name', 'age', 'city'])
 print(df)
```

### 2.4 从CSV文件创建

```python
# 从CSV文件创建
 df = pd.read_csv('data.csv')
 print(df)
```

### 2.5 从Excel文件创建

```python
# 从Excel文件创建
 df = pd.read_excel('data.xlsx')
 print(df)
```

## 第三章：查看数据框

### 3.1 基本信息

```python
# 查看前几行
 print(df.head())

# 查看后几行
 print(df.tail())

# 查看形状
 print(df.shape)

# 查看列名
 print(df.columns)

# 查看索引
 print(df.index)

# 查看数据类型
 print(df.dtypes)

# 查看详细信息
 print(df.info())

# 查看描述性统计
 print(df.describe())
```

### 3.2 数据预览

```python
# 查看前10行
 print(df.head(10))

# 查看后5行
 print(df.tail(5))

# 随机抽样
 print(df.sample(5))
```

## 第四章：访问数据

### 4.1 访问列

```python
# 访问单个列
 print(df['name'])
 print(df.name)  # 仅适用于列名无空格的情况

# 访问多个列
 print(df[['name', 'salary']])
```

### 4.2 访问行

```python
# 通过标签访问行
 print(df.loc[0])
 print(df.loc[1:3])

# 通过位置访问行
 print(df.iloc[0])
 print(df.iloc[1:3])
```

### 4.3 访问单元格

```python
# 通过标签访问单元格
 print(df.loc[0, 'name'])

# 通过位置访问单元格
 print(df.iloc[0, 0])

# 使用at和iat（更快）
 print(df.at[0, 'name'])
 print(df.iat[0, 0])
```

### 4.4 布尔索引

```python
# 布尔索引
 print(df[df['age'] > 30])

# 多条件
 print(df[(df['age'] > 30) & (df['salary'] > 60000)])

# 字符串条件
 print(df[df['city'].str.contains('N')])
```

## 第五章：数据操作

### 5.1 添加和删除列

```python
# 添加列
 df['bonus'] = df['salary'] * 0.1

# 删除列
 df = df.drop('bonus', axis=1)

# 重命名列
 df = df.rename(columns={'salary': 'income'})
```

### 5.2 添加和删除行

```python
# 添加行
 new_row = {'name': 'Frank', 'age': 50, 'city': 'Berlin', 'salary': 100000}
 df = df.append(new_row, ignore_index=True)

# 删除行
 df = df.drop(5, axis=0)

# 重置索引
 df = df.reset_index(drop=True)
```

### 5.3 数据排序

```python
# 按列排序
 df_sorted = df.sort_values('salary', ascending=False)

# 按多列排序
 df_sorted = df.sort_values(['age', 'salary'], ascending=[True, False])

# 按索引排序
 df_sorted = df.sort_index()
```

### 5.4 数据转换

```python
# 应用函数
 df['salary'] = df['salary'].apply(lambda x: x * 1.1)

# 映射
 df['city_code'] = df['city'].map({'New York': 'NY', 'London': 'LN', 'Paris': 'PR'})

# 替换
 df['city'] = df['city'].replace('New York', 'NYC')

# 类型转换
 df['age'] = df['age'].astype(float)
```

## 第六章：数据处理

### 6.1 缺失值处理

```python
# 检测缺失值
 print(df.isna())
 print(df.isna().sum())

# 填充缺失值
 df_filled = df.fillna(0)
 df_filled = df.fillna(df.mean())
 df_filled = df.fillna(method='ffill')  # 向前填充
 df_filled = df.fillna(method='bfill')  # 向后填充

# 丢弃缺失值
 df_dropped = df.dropna()
 df_dropped = df.dropna(axis=1)
```

### 6.2 重复值处理

```python
# 检测重复值
 print(df.duplicated())

# 丢弃重复值
 df_unique = df.drop_duplicates()
 df_unique = df.drop_duplicates(subset=['name'])
```

### 6.3 数据分组

```python
# 按列分组
 grouped = df.groupby('city')

# 分组聚合
 print(grouped['salary'].mean())
 print(grouped['salary'].agg(['mean', 'sum', 'count']))

# 多列分组
 grouped = df.groupby(['city', 'age'])
 print(grouped['salary'].mean())
```

### 6.4 数据合并

```python
# 内连接
 df_merged = pd.merge(df1, df2, on='id', how='inner')

# 左连接
 df_merged = pd.merge(df1, df2, on='id', how='left')

# 右连接
 df_merged = pd.merge(df1, df2, on='id', how='right')

# 外连接
 df_merged = pd.merge(df1, df2, on='id', how='outer')

# 连接
 df_concat = pd.concat([df1, df2])
```

### 6.5 数据透视表

```python
# 创建数据透视表
 pivot_table = pd.pivot_table(df, values='salary', index='city', columns='age', aggfunc=np.mean)

# 多级数据透视表
 pivot_table = pd.pivot_table(df, values='salary', index=['city', 'age'], columns='department', aggfunc=np.sum)
```

## 第七章：时间序列处理

### 7.1 创建时间序列

```python
# 创建日期范围
 dates = pd.date_range('2023-01-01', periods=365, freq='D')
 df = pd.DataFrame({'date': dates, 'value': np.random.randn(365)})

# 设置日期索引
 df = df.set_index('date')
```

### 7.2 时间序列操作

```python
# 重采样
 df_resampled = df.resample('M').sum()

# 移动平均
 df_ma = df.rolling(window=7).mean()

# 差分
 df_diff = df.diff()

# 时间索引访问
 print(df['2023-01'])
 print(df['2023-01-01':'2023-01-31'])
```

## 第八章：性能优化

### 8.1 内存使用优化

```python
# 查看内存使用
 print(df.memory_usage())
 print(df.memory_usage(deep=True))

# 优化数据类型
 df['age'] = df['age'].astype('int32')
 df['salary'] = df['salary'].astype('float32')

# 分类数据类型
 df['city'] = df['city'].astype('category')
```

### 8.2 计算性能优化

```python
# 使用向量化操作
 df['salary'] = df['salary'] * 1.1  # 而不是使用循环

# 使用内置函数
 df['salary'].sum()  # 而不是使用sum(df['salary'])

# 避免链式操作
 # 不好的做法
 df = df[df['age'] > 30]['salary'].mean()
 # 好的做法
 filtered = df[df['age'] > 30]
 mean_salary = filtered['salary'].mean()
```

### 8.3 I/O性能优化

```python
# 读取大文件
 # 分块读取
 chunks = []
 for chunk in pd.read_csv('large_file.csv', chunksize=10000):
     chunks.append(chunk)
 df = pd.concat(chunks)

# 保存文件
 df.to_csv('output.csv', index=False)
 df.to_parquet('output.parquet')  # 更快的文件格式
```

## 第九章：应用示例

### 9.1 数据分析

```python
# 模拟销售数据
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=365, freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], 365),
    'sales': np.random.randint(100, 1000, 365),
    'profit': np.random.randint(10, 100, 365)
})

# 按产品分析
product_analysis = sales_data.groupby('product').agg({
    'sales': ['sum', 'mean', 'std'],
    'profit': ['sum', 'mean', 'std']
})
print(product_analysis)

# 按月分析
sales_data['month'] = sales_data['date'].dt.month
monthly_analysis = sales_data.groupby('month').agg({
    'sales': 'sum',
    'profit': 'sum'
})
print(monthly_analysis)
```

### 9.2 数据清洗

```python
# 创建包含脏数据的示例
dirty_data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, '30', 35, np.nan, 45],
    'salary': [50000, 60000, '70000', 80000, 90000]
})

# 转换数据类型
dirty_data['age'] = pd.to_numeric(dirty_data['age'], errors='coerce')
dirty_data['salary'] = pd.to_numeric(dirty_data['salary'], errors='coerce')

# 填充缺失值
dirty_data['age'] = dirty_data['age'].fillna(dirty_data['age'].mean())

# 验证数据
print(dirty_data)
print(dirty_data.dtypes)
```

### 9.3 数据可视化

```python
import matplotlib.pyplot as plt

# 柱状图
df.plot(kind='bar', x='name', y='salary')
plt.title('Salary Distribution')
plt.xlabel('Name')
plt.ylabel('Salary')
plt.show()

# 折线图
df.plot(kind='line', x='date', y='value')
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# 散点图
df.plot(kind='scatter', x='age', y='salary')
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
```

## 第十章：最佳实践

### 10.1 代码组织

1. **使用链式操作**：利用Pandas的链式操作，使代码更简洁
2. **使用管道操作**：使用`pipe()`方法进行复杂的数据处理
3. **模块化**：将数据处理逻辑封装为函数
4. **注释**：添加清晰的注释，解释代码逻辑

### 10.2 数据处理

1. **先理解数据**：使用`head()`、`info()`等方法了解数据结构
2. **处理缺失值**：根据数据特点选择合适的缺失值处理方法
3. **数据验证**：验证数据的完整性和正确性
4. **数据转换**：根据分析需求转换数据格式

### 10.3 性能优化

1. **使用向量化操作**：避免使用Python循环
2. **选择合适的数据类型**：根据数据特点选择合适的数据类型
3. **内存管理**：监控和优化内存使用
4. **I/O优化**：优化文件读写操作

### 10.4 错误处理

1. **异常处理**：使用try-except处理可能的错误
2. **数据验证**：验证输入数据的有效性
3. **日志记录**：记录数据处理过程中的重要信息

## 第十一章：习题

### 11.1 选择题

1. 以下哪个是创建数据框的正确方法？
   - A) pd.DataFrame([1, 2, 3])
   - B) pd.dataframe({'a': 1, 'b': 2})
   - C) pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
   - D) pd.DataFrame(1, 2, 3)

2. 以下哪个方法用于访问数据框的列？
   - A) df.loc['column']
   - B) df['column']
   - C) df.iloc['column']
   - D) df.column()

3. 以下哪个方法用于处理缺失值？
   - A) dropna()
   - B) drop()
   - C) fill()
   - D) clean()

### 11.2 填空题

1. 数据框的基本结构是________________。
2. 访问数据框行的方法有________________和________________。
3. 处理缺失值的方法有________________、________________和________________。

### 11.3 简答题

1. 简述数据框和NumPy数组的区别。
2. 简述数据框的应用场景。
3. 简述如何优化数据框的性能。

### 11.4 编程题

1. 创建一个包含姓名、年龄、城市和薪资的示例数据框，并计算平均薪资。
2. 读取一个CSV文件，处理其中的缺失值，并计算各列的统计指标。
3. 创建一个时间序列数据框，计算月度汇总和7天移动平均。
4. 合并两个数据框，计算合并后数据框的统计信息。

## 第十二章：总结

### 12.1 知识回顾

1. **数据框的创建**：从字典、NumPy数组、列表、文件等创建数据框
2. **数据访问**：访问列、行和单元格
3. **数据操作**：添加、删除、修改数据
4. **数据处理**：缺失值处理、重复值处理、数据转换
5. **数据分组**：按条件分组并进行聚合
6. **数据合并**：合并多个数据框
7. **时间序列**：处理和分析时间序列数据
8. **性能优化**：优化内存使用和计算速度
9. **应用示例**：数据分析、数据清洗、数据可视化

### 12.2 学习建议

1. **实践练习**：多练习创建和操作数据框
2. **性能测试**：测试不同方法的性能
3. **应用开发**：在实际项目中应用数据框
4. **代码优化**：学习如何优化代码
5. **理论学习**：了解数据框的底层实现

### 12.3 进阶学习

1. 高级数据处理技巧
2. 数据可视化
3. 机器学习中的数据准备
4. 大数据处理
5. 实时数据处理