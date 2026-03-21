"""
Seaborn 进阶演示
Seaborn Advanced Demo
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 设置Seaborn样式
sns.set_style("whitegrid")

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Seaborn 进阶演示")
print("=" * 70)

# 1. 基本绘图
print("\n1. 基本绘图...")

# 生成数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
data = pd.DataFrame({'x': x, 'y': y})

# 创建图形
plt.figure(figsize=(8, 6))
sns.lineplot(x='x', y='y', data=data)
plt.title('Seaborn 折线图')
plt.xlabel('x')
plt.ylabel('y')

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'basic_seaborn_plot.png'))
print("Seaborn 折线图已保存为 'images/basic_seaborn_plot.png'")

# 2. 分类图
print("\n2. 分类图...")

# 创建分类数据
category_data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'] * 20,
    'value': np.random.randn(100) + np.repeat([1, 2, 3, 4, 5], 20)
})

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=category_data)
plt.title('箱线图')
plt.savefig(os.path.join(images_dir, 'boxplot.png'))
print("箱线图已保存为 'images/boxplot.png'")

# 小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', data=category_data)
plt.title('小提琴图')
plt.savefig(os.path.join(images_dir, 'violinplot.png'))
print("小提琴图已保存为 'images/violinplot.png'")

# 条形图
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='value', data=category_data)
plt.title('条形图')
plt.savefig(os.path.join(images_dir, 'barplot.png'))
print("条形图已保存为 'images/barplot.png'")

# 3. 分布分析
print("\n3. 分布分析...")

# 生成分布数据
dist_data = pd.DataFrame({
    'normal': np.random.normal(0, 1, 1000),
    'uniform': np.random.uniform(-3, 3, 1000),
    'exponential': np.random.exponential(1, 1000)
})

# 直方图
plt.figure(figsize=(10, 6))
sns.histplot(dist_data['normal'], kde=True)
plt.title('直方图')
plt.savefig(os.path.join(images_dir, 'histplot.png'))
print("直方图已保存为 'images/histplot.png'")

# 密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(dist_data['normal'], fill=True)
plt.title('密度图')
plt.savefig(os.path.join(images_dir, 'kdeplot.png'))
print("密度图已保存为 'images/kdeplot.png'")

# 多变量分布
plt.figure(figsize=(10, 6))
sns.jointplot(x='normal', y='uniform', data=dist_data, kind='scatter')
plt.savefig(os.path.join(images_dir, 'jointplot.png'))
print("联合分布图已保存为 'images/jointplot.png'")

# 4. 关系图
print("\n4. 关系图...")

# 生成关系数据
relationship_data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100) + 0.5 * np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# 散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='x', y='y', hue='category', data=relationship_data)
plt.title('散点图')
plt.savefig(os.path.join(images_dir, 'scatterplot.png'))
print("散点图已保存为 'images/scatterplot.png'")

# 回归图
plt.figure(figsize=(10, 6))
sns.regplot(x='x', y='y', data=relationship_data)
plt.title('回归图')
plt.savefig(os.path.join(images_dir, 'regplot.png'))
print("回归图已保存为 'images/regplot.png'")

# 5. 热力图
print("\n5. 热力图...")

# 生成热力图数据
corr_data = np.random.randn(10, 10)
corr_matrix = np.corrcoef(corr_data)

# 热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('热力图')
plt.savefig(os.path.join(images_dir, 'heatmap.png'))
print("热力图已保存为 'images/heatmap.png'")

# 6. 多子图
print("\n6. 多子图...")

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
plt.savefig(os.path.join(images_dir, 'subplots.png'))
print("多子图已保存为 'images/subplots.png'")

# 7. 样式和主题
print("\n7. 样式和主题...")

# 不同样式
styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']

plt.figure(figsize=(14, 10))

for i, style in enumerate(styles):
    sns.set_style(style)
    ax = plt.subplot(2, 3, i + 1)
    sns.lineplot(x='x', y='y', data=data)
    ax.set_title(f'Style: {style}')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'styles.png'))
print("样式示例已保存为 'images/styles.png'")

# 8. 颜色主题
print("\n8. 颜色主题...")

# 不同颜色主题
palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']

plt.figure(figsize=(14, 10))

for i, palette in enumerate(palettes):
    ax = plt.subplot(2, 3, i + 1)
    sns.set_palette(palette)
    sns.barplot(x='category', y='value', data=category_data)
    ax.set_title(f'Palette: {palette}')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'palettes.png'))
print("颜色主题示例已保存为 'images/palettes.png'")

# 9. 应用示例
print("\n9. 应用示例...")

# 9.1 数据可视化
print("\n9.1 数据可视化...")

# 模拟销售数据
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
plt.savefig(os.path.join(images_dir, 'sales_trend.png'))
print("销售趋势图已保存为 'images/sales_trend.png'")

# 9.2 多变量分析
print("\n9.2 多变量分析...")

# 模拟多变量数据
multi_data = pd.DataFrame({
    'age': np.random.randint(18, 70, 100),
    'income': np.random.randint(20000, 100000, 100),
    'expense': np.random.randint(10000, 80000, 100),
    'gender': np.random.choice(['Male', 'Female'], 100)
})

# 多变量关系
plt.figure(figsize=(12, 6))
sns.pairplot(multi_data, hue='gender')
plt.savefig(os.path.join(images_dir, 'pairplot.png'))
print("多变量关系图已保存为 'images/pairplot.png'")

# 10. 性能测试
print("\n10. 性能测试...")
import time

# 测试不同大小数据的绘图性能
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

# 11. 总结
print("\n" + "=" * 70)
print("Seaborn 进阶总结")
print("=" * 70)

print("""
Seaborn 功能：

1. 基本绘图：
   - 折线图
   - 散点图
   - 条形图

2. 分类图：
   - 箱线图
   - 小提琴图
   - 条形图
   - 点图

3. 分布分析：
   - 直方图
   - 密度图
   - 联合分布图
   - 成对关系图

4. 关系图：
   - 散点图
   - 回归图
   - 线图

5. 热力图：
   - 相关性热力图
   - 混淆矩阵热力图

6. 样式和主题：
   - 不同样式（white, dark, whitegrid, darkgrid, ticks）
   - 不同颜色主题（deep, muted, bright, pastel, dark, colorblind）

7. 应用场景：
   - 数据可视化
   - 统计分析
   - 机器学习数据探索
   - 业务分析

8. 性能考虑：
   - 大型数据的绘图性能
   - 内存使用
   - 渲染速度
""")

print("=" * 70)
print("Seaborn 进阶演示完成！")
print("=" * 70)