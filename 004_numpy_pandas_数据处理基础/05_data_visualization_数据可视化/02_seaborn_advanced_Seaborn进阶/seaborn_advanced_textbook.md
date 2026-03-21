# Seaborn 进阶教材

## 第一章：Seaborn 简介

### 1.1 什么是 Seaborn

Seaborn 是基于 Matplotlib 的高级数据可视化库，它提供了更美观、更简洁的接口来创建统计图表。Seaborn 内置了多种主题和颜色方案，使得创建专业的可视化变得更加容易。

### 1.2 Seaborn 的特点

- **美观的默认样式**：提供了现代化、美观的默认样式
- **高级统计图表**：内置了多种统计图表，如箱线图、小提琴图、热力图等
- **简化的接口**：相比 Matplotlib，接口更加简洁易用
- **与 Pandas 集成**：直接支持 Pandas DataFrame
- **多种颜色主题**：提供了多种预设的颜色主题

### 1.3 Seaborn 的应用场景

- **统计分析**：创建各种统计图表
- **数据探索**：快速了解数据分布和关系
- **机器学习**：数据预处理和特征可视化
- **业务分析**：创建专业的业务报表
- **科学研究**：展示实验数据和结果

## 第二章：基本绘图

### 2.1 折线图

折线图用于展示数据随时间或其他连续变量的变化趋势。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 生成数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
data = pd.DataFrame({'x': x, 'y': y})

# 创建折线图
plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='y', data=data)
plt.title('Seaborn 折线图')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 2.2 散点图

散点图用于展示两个变量之间的关系。

```python
# 生成散点图数据
relationship_data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100) + 0.5 * np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# 创建散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='category', data=relationship_data)
plt.title('散点图')
plt.show()
```

### 2.3 条形图

条形图用于比较不同类别的数据值。

```python
# 生成条形图数据
category_data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'] * 20,
    'value': np.random.randn(100) + np.repeat([1, 2, 3, 4, 5], 20)
})

# 创建条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='value', data=category_data)
plt.title('条形图')
plt.show()
```

## 第三章：分类图

### 3.1 箱线图

箱线图用于展示数据的分布情况，包括中位数、四分位数和异常值。

```python
# 创建箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=category_data)
plt.title('箱线图')
plt.show()
```

### 3.2 小提琴图

小提琴图结合了箱线图和密度图的特点，展示数据的分布情况。

```python
# 创建小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', data=category_data)
plt.title('小提琴图')
plt.show()
```

### 3.3 点图

点图用于展示不同类别的数据点位置。

```python
# 创建点图
plt.figure(figsize=(10, 6))
sns.pointplot(x='category', y='value', data=category_data)
plt.title('点图')
plt.show()
```

## 第四章：分布分析

### 4.1 直方图

直方图用于展示数据的分布情况。

```python
# 生成分布数据
dist_data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'uniform': np.random.uniform(-3, 3, 1000),
    'exponential': np.random.exponential(1, 1000)
})

# 创建直方图
plt.figure(figsize=(10, 6))
sns.histplot(dist_data['normal'], kde=True)
plt.title('直方图')
plt.show()
```

### 4.2 密度图

密度图用于展示数据的概率密度函数。

```python
# 创建密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(dist_data['normal'], fill=True)
plt.title('密度图')
plt.show()
```

### 4.3 联合分布图

联合分布图用于展示两个变量之间的关系。

```python
# 创建联合分布图
plt.figure(figsize=(10, 6))
sns.jointplot(x='normal', y='uniform', data=dist_data, kind='scatter')
plt.title('联合分布图')
plt.show()
```

### 4.4 成对关系图

成对关系图用于展示多个变量之间的关系。

```python
# 创建成对关系图
sns.pairplot(dist_data)
plt.title('成对关系图')
plt.show()
```

## 第五章：关系图

### 5.1 回归图

回归图用于展示两个变量之间的线性关系。

```python
# 创建回归图
plt.figure(figsize=(10, 6))
sns.regplot(x='x', y='y', data=relationship_data)
plt.title('回归图')
plt.show()
```

### 5.2 残差图

残差图用于展示回归模型的残差分布。

```python
# 创建残差图
plt.figure(figsize=(10, 6))
sns.residplot(x='x', y='y', data=relationship_data)
plt.title('残差图')
plt.show()
```

## 第六章：热力图

热力图用于展示矩阵数据的强度。

```python
# 生成热力图数据
corr_data = np.random.randn(10, 10)
corr_matrix = np.corrcoef(corr_data)

# 创建热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('热力图')
plt.show()
```

## 第七章：多子图

### 7.1 基本子图

```python
# 创建多子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 第一个子图
sns.histplot(dist_data['normal'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('正态分布')

# 第二个子图
sns.scatterplot(x='x', y='y', hue='category', data=relationship_data, ax=axes[0, 1])
axes[0, 1].set_title('散点图')

# 第三个子图
sns.boxplot(x='category', y='value', data=category_data, ax=axes[1, 0])
axes[1, 0].set_title('箱线图')

# 第四个子图
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('热力图')

plt.tight_layout()
plt.show()
```

### 7.2 FacetGrid

FacetGrid 用于创建多子图网格。

```python
# 创建 FacetGrid
g = sns.FacetGrid(relationship_data, col='category')
g.map(sns.scatterplot, 'x', 'y')
plt.show()
```

## 第八章：样式和主题

### 8.1 样式设置

Seaborn 提供了多种预设样式。

```python
# 设置样式
sns.set_style("whitegrid")  # 其他选项：'white', 'dark', 'darkgrid', 'ticks'

# 创建图表
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data)
plt.title('白色网格样式')
plt.show()
```

### 8.2 颜色主题

Seaborn 提供了多种预设颜色主题。

```python
# 设置颜色主题
sns.set_palette("deep")  # 其他选项：'muted', 'bright', 'pastel', 'dark', 'colorblind'

# 创建图表
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='value', data=category_data)
plt.title('Deep 颜色主题')
plt.show()
```

### 8.3 上下文设置

上下文设置用于调整图表的比例和字体大小。

```python
# 设置上下文
sns.set_context("paper")  # 其他选项：'notebook', 'talk', 'poster'

# 创建图表
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=data)
plt.title('Paper 上下文')
plt.show()
```

## 第九章：性能优化

### 9.1 大型数据的绘图

当处理大型数据集时，需要考虑绘图性能。

```python
# 测试不同大小数据的绘图性能
import time

sizes = [100, 1000, 10000, 100000]
times = []

for size in sizes:
    test_data = pd.DataFrame({
        'x': np.linspace(0, 10, size),
        'y': np.sin(np.linspace(0, 10, size)) + np.random.normal(0, 0.1, size)
    })
    
    start = time.time()
    plt.figure()
    sns.lineplot(x='x', y='y', data=test_data)
    plt.close()
    end = time.time()
    
    times.append(end - start)
    print(f"绘制 {size} 个点耗时: {end - start:.6f}秒")
```

### 9.2 内存优化

对于大型数据集，还需要考虑内存使用。

```python
# 内存优化
# 1. 减少数据点数量
# 对于大型数据集，可以考虑采样

# 2. 使用适当的数据类型
# 对于数值数据，使用适当的数值类型

# 3. 避免不必要的计算
# 预处理数据，减少绘图时的计算
```

## 第十章：应用示例

### 10.1 销售趋势分析

```python
# 销售趋势分析
sales_data = pd.DataFrame({
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'sales': [1200, 1900, 1500, 1600, 2100, 2500, 2300, 2400, 2800, 3100, 3300, 3500],
    'profit': [120, 190, 150, 160, 210, 250, 230, 240, 280, 310, 330, 350]
})

# 销售趋势图
plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='sales', data=sales_data, marker='o', linewidth=2)
plt.title('月度销售趋势')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.show()
```

### 10.2 多变量分析

```python
# 多变量分析
multi_data = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 100000, 100),
    'expense': np.random.randint(10000, 80000, 100),
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# 多变量关系
plt.figure(figsize=(12, 6))
sns.pairplot(multi_data, hue='gender')
plt.title('多变量关系图')
plt.show()
```

### 10.3 相关性分析

```python
# 相关性分析
corr = multi_data.corr()

# 相关性热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('相关性热力图')
plt.show()
```

## 第十一章：最佳实践

### 11.1 图表设计

1. **选择合适的图表类型**：根据数据特点选择合适的图表类型
2. **使用适当的样式**：选择适合数据的样式和颜色主题
3. **保持简洁**：避免图表过于复杂，突出重点
4. **添加必要的标签**：确保图表有清晰的标题、坐标轴标签和图例
5. **优化布局**：使用适当的布局和尺寸

### 11.2 代码组织

1. **模块化**：将绘图逻辑封装为函数
2. **注释**：添加清晰的注释，解释代码逻辑
3. **异常处理**：处理可能的错误情况
4. **代码风格**：遵循 Python 代码风格规范

### 11.3 性能优化

1. **数据处理**：在绘图前处理数据，减少绘图时的计算
2. **内存管理**：对于大型数据集，考虑内存使用
3. **渲染优化**：使用适当的渲染方法，提高绘图速度
4. **缓存**：对于重复绘制的图表，考虑使用缓存

### 11.4 输出格式

1. **选择合适的格式**：根据需要选择合适的输出格式
2. **分辨率**：设置适当的分辨率，确保图表清晰
3. **大小**：设置适当的图表大小，确保内容完整
4. **文件命名**：使用清晰、描述性的文件名

## 第十二章：习题

### 12.1 选择题

1. 以下哪个函数用于创建箱线图？
   - A) sns.boxplot()
   - B) sns.violinplot()
   - C) sns.barplot()
   - D) sns.scatterplot()

2. 以下哪个函数用于创建热力图？
   - A) sns.heatmap()
   - B) sns.heatmap()
   - C) sns.heatmap()
   - D) sns.heatmap()

3. 以下哪个参数用于设置 Seaborn 样式？
   - A) sns.set_style()
   - B) sns.set_palette()
   - C) sns.set_context()
   - D) sns.set()

### 12.2 填空题

1. Seaborn 是基于 ________________ 构建的高级数据可视化库。
2. 用于创建散点图的函数是 ________________。
3. 用于设置颜色主题的函数是 ________________。

### 12.3 简答题

1. 简述 Seaborn 与 Matplotlib 的区别。
2. 简述如何选择合适的图表类型。
3. 简述如何优化 Seaborn 图表的性能。

### 12.4 编程题

1. 创建一个包含 100 个点的正弦曲线，并使用 Seaborn 绘制折线图。
2. 创建一个箱线图，展示 5 个类别的数据分布。
3. 创建一个热力图，展示 10x10 随机矩阵的相关性。
4. 创建一个多子图，分别展示直方图、散点图、箱线图和热力图。

## 第十三章：总结

### 13.1 知识回顾

1. **Seaborn 基础**：Seaborn 的基本概念和功能
2. **基本图表类型**：折线图、散点图、条形图
3. **分类图**：箱线图、小提琴图、点图
4. **分布分析**：直方图、密度图、联合分布图、成对关系图
5. **关系图**：回归图、残差图
6. **热力图**：相关性热力图、混淆矩阵热力图
7. **多子图**：基本子图、FacetGrid
8. **样式和主题**：多种预设样式和颜色主题
9. **性能优化**：大型数据处理、内存优化
10. **应用示例**：销售趋势分析、多变量分析、相关性分析

### 13.2 学习建议

1. **实践练习**：多练习创建不同类型的图表
2. **真实数据**：使用真实的数据进行可视化
3. **工具探索**：探索 Seaborn 的高级功能
4. **其他库**：学习其他可视化库，如 Plotly、Bokeh 等
5. **领域应用**：将可视化技术应用到特定领域

### 13.3 进阶学习

1. **Seaborn 高级特性**：自定义主题、高级图表类型
2. **与其他库的集成**：与 Pandas、NumPy 的集成
3. **交互式可视化**：结合 Plotly 或 Bokeh 创建交互式图表
4. **自定义扩展**：创建自定义图表类型
5. **实时数据可视化**：处理和可视化实时数据