"""
Matplotlib 基础演示
Matplotlib Basics Demo
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Matplotlib 基础演示")
print("=" * 70)

# 1. 基本绘图
print("\n1. 基本绘图...")

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

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'basic_plot.png'))
print("基本折线图已保存为 'images/basic_plot.png'")

# 2. 多种图表类型
print("\n2. 多种图表类型...")

# 柱状图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 柱状图
data = [10, 20, 30, 40, 50]
labels = ['A', 'B', 'C', 'D', 'E']
ax = axes[0, 0]
ax.bar(labels, data, color='steelblue')
ax.set_title('柱状图')
ax.set_xlabel('类别')
ax.set_ylabel('值')
ax.grid(True, alpha=0.3, axis='y')

# 散点图
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)
ax = axes[0, 1]
scatter = ax.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5)
ax.set_title('散点图')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter, ax=ax)

# 直方图
data_hist = np.random.randn(1000)
ax = axes[1, 0]
ax.hist(data_hist, bins=30, color='coral', alpha=0.7)
ax.set_title('直方图')
ax.set_xlabel('值')
ax.set_ylabel('频率')
ax.grid(True, alpha=0.3, axis='y')

# 饼图
pie_data = [30, 20, 25, 15, 10]
pie_labels = ['A', 'B', 'C', 'D', 'E']
ax = axes[1, 1]
ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
ax.set_title('饼图')
ax.axis('equal')  # 确保饼图是圆的

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'multiple_plots.png'))
print("多种图表类型已保存为 'images/multiple_plots.png'")

# 3. 自定义样式
print("\n3. 自定义样式...")

# 自定义线条样式
plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)', color='red', linestyle='-', linewidth=2, marker='o', markersize=5)
plt.plot(x, np.cos(x), label='cos(x)', color='blue', linestyle='--', linewidth=2, marker='s', markersize=5)
plt.title('自定义线条样式')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(images_dir, 'custom_styles.png'))
print("自定义样式已保存为 'images/custom_styles.png'")

# 4. 子图布局
print("\n4. 子图布局...")

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
plt.savefig(os.path.join(images_dir, 'subplot_grid.png'))
print("子图布局已保存为 'images/subplot_grid.png'")

# 5. 文本和注释
print("\n5. 文本和注释...")

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)')
plt.title('带有文本和注释的图表')
plt.xlabel('x')
plt.ylabel('y')

# 添加文本
plt.text(0, 0, '原点', fontsize=12, color='red')

# 添加注释
plt.annotate('最大值', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('最小值', xy=(3*np.pi/2, -1), xytext=(3*np.pi/2 + 1, -0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.legend()
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'text_annotations.png'))
print("文本和注释已保存为 'images/text_annotations.png'")

# 6. 颜色和样式
print("\n6. 颜色和样式...")

# 颜色映射
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='值')
plt.title('颜色映射示例')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(os.path.join(images_dir, 'color_mapping.png'))
print("颜色映射已保存为 'images/color_mapping.png'")

# 7. 3D 图表
print("\n7. 3D 图表...")

try:
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
    
    plt.savefig(os.path.join(images_dir, '3d_plot.png'))
    print("3D 图表已保存为 'images/3d_plot.png'")
except ImportError:
    print("3D 图表功能不可用，跳过")

# 8. 性能测试
print("\n8. 性能测试...")
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

# 9. 应用示例
print("\n9. 应用示例...")

# 9.1 数据可视化
print("\n9.1 数据可视化...")

# 模拟销售数据
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

plt.savefig(os.path.join(images_dir, 'sales_trend.png'))
print("销售趋势图已保存为 'images/sales_trend.png'")

# 9.2 多数据系列
print("\n9.2 多数据系列...")

# 模拟多个产品的销售数据
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
plt.savefig(os.path.join(images_dir, 'multiple_products.png'))
print("多产品销售趋势图已保存为 'images/multiple_products.png'")

# 10. 总结
print("\n" + "=" * 70)
print("Matplotlib 基础总结")
print("=" * 70)

print("""
Matplotlib 功能：

1. 基本绘图：
   - 折线图
   - 柱状图
   - 散点图
   - 直方图
   - 饼图

2. 自定义样式：
   - 线条样式
   - 颜色
   - 标记
   - 图例
   - 网格

3. 子图布局：
   - 网格布局
   - 自定义布局
   - 3D 图表

4. 文本和注释：
   - 标题
   - 坐标轴标签
   - 文本
   - 注释

5. 颜色和样式：
   - 颜色映射
   - 样式表
   - 自定义颜色

6. 应用场景：
   - 数据可视化
   - 科学研究
   - 业务分析
   - 报告生成

7. 性能考虑：
   - 大型数据的绘图性能
   - 内存使用
   - 渲染速度
""")

print("=" * 70)
print("Matplotlib 基础演示完成！")
print("=" * 70)