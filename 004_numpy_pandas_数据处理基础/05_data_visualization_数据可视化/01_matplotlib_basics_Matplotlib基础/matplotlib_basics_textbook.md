# Matplotlib 基础教材

## 第一章：Matplotlib 简介

### 1.1 什么是 Matplotlib

Matplotlib 是 Python 中最常用的数据可视化库之一，它提供了丰富的绘图功能，可以创建各种类型的图表，如折线图、柱状图、散点图、直方图等。Matplotlib 适用于数据可视化、科学研究、业务分析等多种场景。

### 1.2 Matplotlib 的特点

- **功能丰富**：支持多种图表类型和自定义选项
- **高度可定制**：几乎可以自定义图表的每一个元素
- **与 NumPy 集成**：直接支持 NumPy 数组
- **跨平台**：可以在不同操作系统上使用
- **支持多种输出格式**：可以保存为 PNG、PDF、SVG 等格式

### 1.3 Matplotlib 的应用场景

- **数据可视化**：展示数据趋势和模式
- **科学研究**：绘制实验数据和结果
- **业务分析**：创建销售报表和财务图表
- **教育教学**：展示数学概念和统计数据
- **报告生成**：生成高质量的图表用于报告和论文

## 第二章：基本绘图

### 2.1 折线图

折线图是最基本的图表类型，用于展示数据随时间或其他连续变量的变化趋势。

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 创建图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.title('基本折线图')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 2.2 柱状图

柱状图用于比较不同类别的数据值。

```python
# 柱状图数据
data = [10, 20, 30, 40, 50]
labels = ['A', 'B', 'C', 'D', 'E']

plt.figure(figsize=(8, 6))
plt.bar(labels, data, color='steelblue')
plt.title('柱状图')
plt.xlabel('类别')
plt.ylabel('值')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

### 2.3 散点图

散点图用于展示两个变量之间的关系。

```python
# 散点图数据
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5)
plt.title('散点图')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='颜色')
plt.show()
```

### 2.4 直方图

直方图用于展示数据的分布情况。

```python
# 直方图数据
data_hist = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(data_hist, bins=30, color='coral', alpha=0.7)
plt.title('直方图')
plt.xlabel('值')
plt.ylabel('频率')
plt.grid(True, alpha=0.3, axis='y')
plt.show()
```

### 2.5 饼图

饼图用于展示部分与整体的关系。

```python
# 饼图数据
pie_data = [30, 20, 25, 15, 10]
pie_labels = ['A', 'B', 'C', 'D', 'E']

plt.figure(figsize=(8, 6))
plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
plt.title('饼图')
plt.axis('equal')  # 确保饼图是圆的
plt.show()
```

## 第三章：自定义样式

### 3.1 线条样式

Matplotlib 提供了丰富的线条样式选项，可以自定义线条的颜色、线型、宽度和标记。

```python
# 自定义线条样式
x = np.linspace(0, 10, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='red', linestyle='-', linewidth=2, marker='o', markersize=5)
plt.plot(x, np.cos(x), label='cos(x)', color='blue', linestyle='--', linewidth=2, marker='s', markersize=5)
plt.title('自定义线条样式')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

### 3.2 颜色设置

Matplotlib 支持多种颜色设置方式，包括命名颜色、RGB 值、十六进制颜色代码等。

```python
# 颜色设置
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), color='red', label='红色')
plt.plot(x, np.cos(x), color='blue', label='蓝色')
plt.plot(x, np.tan(x), color='green', label='绿色')
plt.plot(x, np.exp(x), color='purple', label='紫色')
plt.title('颜色设置')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

### 3.3 标记样式

标记用于标识数据点的位置，Matplotlib 提供了多种标记样式。

```python
# 标记样式
plt.figure(figsize=(10, 6))
plt.plot(x[:20], np.sin(x[:20]), marker='o', label='圆形')
plt.plot(x[:20], np.cos(x[:20]), marker='s', label='方形')
plt.plot(x[:20], np.tan(x[:20]), marker='^', label='三角形')
plt.plot(x[:20], np.exp(x[:20]), marker='*', label='星形')
plt.title('标记样式')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

## 第四章：子图布局

### 4.1 网格布局

网格布局是创建多个子图的常用方法，可以将图形区域划分为规则的网格。

```python
# 网格布局
fig = plt.figure(figsize=(14, 10))

# 2x2网格
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, np.sin(x))
ax1.set_title('sin(x)')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, np.cos(x))
ax2.set_title('cos(x)')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, np.tan(x))
ax3.set_title('tan(x)')
ax3.set_ylim(-10, 10)  # 设置y轴范围

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, np.exp(x))
ax4.set_title('exp(x)')
ax4.set_yscale('log')  # 设置y轴为对数刻度

plt.tight_layout()
plt.show()
```

### 4.2 自定义布局

除了网格布局，Matplotlib 还支持更灵活的自定义布局。

```python
# 自定义布局
fig = plt.figure(figsize=(14, 10))

# 第一行一个大图
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x, np.sin(x))
ax1.set_title('sin(x)')

# 第二行两个小图
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(x, np.cos(x))
ax2.set_title('cos(x)')

ax3 = fig.add_subplot(2, 2, 4)
ax3.plot(x, np.tan(x))
ax3.set_title('tan(x)')
ax3.set_ylim(-10, 10)

plt.tight_layout()
plt.show()
```

## 第五章：文本和注释

### 5.1 基本文本

Matplotlib 允许在图表中添加文本，包括标题、坐标轴标签和自定义文本。

```python
# 基本文本
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)')
plt.title('带有文本的图表')
plt.xlabel('x')
plt.ylabel('y')

# 添加文本
plt.text(0, 0, '原点', fontsize=12, color='red')
plt.text(np.pi, 0, 'π', fontsize=12, color='blue')
plt.text(2*np.pi, 0, '2π', fontsize=12, color='green')

plt.legend()
plt.grid(True)
plt.show()
```

### 5.2 注释

注释用于在图表中添加说明性文字，并可以指向特定的数据点。

```python
# 注释
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)')
plt.title('带有注释的图表')
plt.xlabel('x')
plt.ylabel('y')

# 添加注释
plt.annotate('最大值', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('最小值', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 + 1, -0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()
plt.grid(True)
plt.show()
```

## 第六章：颜色和样式

### 6.1 颜色映射

颜色映射用于将数据值映射到颜色，常用于热力图、等高线图等。

```python
# 颜色映射
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.figure(figsize=(10, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='值')
plt.title('颜色映射示例')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

### 6.2 样式表

Matplotlib 提供了多种预定义的样式表，可以快速改变图表的外观。

```python
# 样式表
plt.style.use('seaborn')  # 使用seaborn样式

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.title('使用样式表')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

## 第七章：3D 图表

### 7.1 3D 线图

Matplotlib 支持创建 3D 图表，可以展示三维数据。

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成3D数据
theta = np.linspace(0, 2*np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax.plot(x, y, z, label='3D 螺旋线')
ax.set_title('3D 图表示例')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
```

### 7.2 3D 散点图

3D 散点图用于展示三维空间中的数据点。

```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 生成3D散点数据
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)
colors = np.random.rand(50)
sizes = 100 * np.random.rand(50)

ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.5)
ax.set_title('3D 散点图')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
```

## 第八章：性能优化

### 8.1 大型数据的绘图

当处理大型数据集时，需要考虑绘图性能。

```python
# 大型数据的绘图
import time

# 测试不同大小数据的绘图性能
sizes = [100, 1000, 10000, 100000]
times = []

for size in sizes:
    x = np.linspace(0, 10, size)
    y = np.sin(x)
    
    start = time.time()
    plt.figure()
    plt.plot(x, y)
    plt.close()
    end = time.time()
    
    times.append(end - start)
    print(f"绘制 {size} 个点耗时: {end - start:.6f}秒")

# 绘制性能对比
plt.figure(figsize=(10, 6))
plt.plot(sizes, times, marker='o')
plt.title('绘图性能')
plt.xlabel('数据点数量')
plt.ylabel('时间(秒)')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()
```

### 8.2 内存优化

对于大型数据集，还需要考虑内存使用。

```python
# 内存优化
# 1. 使用适当的数据类型
x = np.linspace(0, 10, 1000000, dtype=np.float32)  # 使用float32减少内存使用

# 2. 避免不必要的中间对象
# 直接绘制，不创建中间变量
plt.figure()
plt.plot(np.linspace(0, 10, 1000000), np.sin(np.linspace(0, 10, 1000000)))
plt.close()

# 3. 批量处理
# 对于非常大的数据集，可以考虑分块处理
```

## 第九章：应用示例

### 9.1 销售趋势分析

```python
# 销售趋势分析
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
sales = [1200, 1900, 1500, 1600, 2100, 2500, 2300, 2400, 2800, 3100, 3300, 3500]

plt.figure(figsize=(12, 6))
plt.plot(months, sales, marker='o', linestyle='-', linewidth=2, color='steelblue')
plt.title('月度销售趋势')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.grid(True, alpha=0.3)

# 添加数据标签
for i, (month, sale) in enumerate(zip(months, sales)):
    plt.text(i, sale + 50, f'{sale}', ha='center')

plt.show()
```

### 9.2 多产品对比

```python
# 多产品对比
product1 = [1200, 1900, 1500, 1600, 2100, 2500, 2300, 2400, 2800, 3100, 3300, 3500]
product2 = [800, 1200, 1000, 1100, 1500, 1800, 1700, 1900, 2200, 2500, 2700, 3000]
product3 = [500, 700, 600, 800, 1000, 1200, 1100, 1300, 1500, 1800, 2000, 2200]

plt.figure(figsize=(12, 6))
plt.plot(months, product1, marker='o', label='产品A', linewidth=2)
plt.plot(months, product2, marker='s', label='产品B', linewidth=2)
plt.plot(months, product3, marker='^', label='产品C', linewidth=2)
plt.title('多产品销售趋势')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 9.3 市场份额分析

```python
# 市场份额分析
market_share = [35, 25, 20, 15, 5]
companies = ['公司A', '公司B', '公司C', '公司D', '其他']
colors = ['steelblue', 'coral', 'green', 'purple', 'gray']

plt.figure(figsize=(8, 8))
plt.pie(market_share, labels=companies, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('市场份额分析')
plt.axis('equal')
plt.show()
```

## 第十章：最佳实践

### 10.1 图表设计

1. **选择合适的图表类型**：根据数据特点选择合适的图表类型
2. **保持简洁**：避免图表过于复杂，突出重点
3. **使用适当的颜色**：选择清晰、对比明显的颜色
4. **添加必要的标签**：确保图表有清晰的标题、坐标轴标签和图例
5. **优化布局**：使用适当的布局和尺寸

### 10.2 代码组织

1. **模块化**：将绘图逻辑封装为函数
2. **注释**：添加清晰的注释，解释代码逻辑
3. **异常处理**：处理可能的错误情况
4. **代码风格**：遵循 Python 代码风格规范

### 10.3 性能优化

1. **数据处理**：在绘图前处理数据，减少绘图时的计算
2. **内存管理**：对于大型数据集，考虑内存使用
3. **渲染优化**：使用适当的渲染方法，提高绘图速度
4. **缓存**：对于重复绘制的图表，考虑使用缓存

### 10.4 输出格式

1. **选择合适的格式**：根据需要选择合适的输出格式
2. **分辨率**：设置适当的分辨率，确保图表清晰
3. **大小**：设置适当的图表大小，确保内容完整
4. **文件命名**：使用清晰、描述性的文件名

## 第十一章：习题

### 11.1 选择题

1. 以下哪个函数用于创建折线图？
   - A) plt.bar()
   - B) plt.scatter()
   - C) plt.plot()
   - D) plt.hist()

2. 以下哪个参数用于设置图表标题？
   - A) title()
   - B) set_title()
   - C) plt.title()
   - D) all of the above

3. 以下哪个方法用于创建子图？
   - A) plt.subplot()
   - B) fig.add_subplot()
   - C) plt.figure()
   - D) both A and B

### 11.2 填空题

1. Matplotlib 中，创建图形的函数是________________。
2. 用于设置 x 轴标签的函数是________________。
3. 用于保存图表的函数是________________。

### 11.3 简答题

1. 简述 Matplotlib 的主要功能。
2. 简述如何自定义图表的样式。
3. 简述如何创建多个子图。

### 11.4 编程题

1. 创建一个包含 100 个点的正弦曲线，并添加标题、坐标轴标签和图例。
2. 创建一个柱状图，展示 5 个类别的数据，并添加数据标签。
3. 创建一个散点图，展示 100 个随机点，并使用颜色和大小表示额外的维度。
4. 创建一个包含 2x2 子图的图形，分别展示正弦、余弦、正切和指数函数。

## 第十二章：总结

### 12.1 知识回顾

1. **Matplotlib 基础**：Matplotlib 的基本概念和功能
2. **基本图表类型**：折线图、柱状图、散点图、直方图、饼图
3. **自定义样式**：线条样式、颜色、标记、图例、网格
4. **布局管理**：子图、网格布局、自定义布局
5. **文本和注释**：标题、标签、文本、注释
6. **颜色和样式**：颜色映射、样式表、自定义颜色
7. **3D 图表**：3D 线图、3D 散点图
8. **性能优化**：大型数据处理、内存优化
9. **应用示例**：销售趋势分析、多产品对比、市场份额分析

### 12.2 学习建议

1. **实践练习**：多练习创建不同类型的图表
2. **真实数据**：使用真实的数据进行可视化
3. **工具探索**：探索 Matplotlib 的高级功能
4. **其他库**：学习其他可视化库，如 Seaborn、Plotly 等
5. **领域应用**：将可视化技术应用到特定领域

### 12.3 进阶学习

1. **Seaborn**：基于 Matplotlib 的高级可视化库
2. **Plotly**：交互式可视化库
3. **Bokeh**：交互式网页可视化库
4. **Altair**：声明式可视化库
5. **Matplotlib 高级特性**：动画、事件处理、自定义艺术家