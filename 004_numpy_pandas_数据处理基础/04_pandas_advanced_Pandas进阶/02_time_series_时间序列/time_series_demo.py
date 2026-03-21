"""
时间序列 (Time Series) 演示
Time Series Demo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("时间序列 (Time Series) 演示")
print("=" * 70)

# 1. 创建时间序列数据
print("\n1. 创建时间序列数据...")

# 从日期范围创建
print("从日期范围创建:")
dates = pd.date_range('2023-01-01', periods=365, freq='D')
data = np.random.randn(365).cumsum()
ts = pd.Series(data, index=dates)
print(ts.head())

# 从DataFrame创建
print("\n从DataFrame创建:")
df = pd.DataFrame({
    'date': dates,
    'value': data
})
df.set_index('date', inplace=True)
print(df.head())

# 2. 时间序列索引
print("\n2. 时间序列索引...")

# 按日期访问
print("按日期访问:")
print(ts['2023-01-01'])
print(ts['2023-01-01':'2023-01-05'])

# 按月份访问
print("\n按月份访问:")
print(ts['2023-01'])

# 按年份访问
print("\n按年份访问:")
print(ts['2023'])

# 3. 时间序列属性
print("\n3. 时间序列属性...")

print("索引类型:", type(ts.index))
print("索引频率:", ts.index.freq)
print("开始日期:", ts.index.min())
print("结束日期:", ts.index.max())
print("时间范围:", ts.index.max() - ts.index.min())

# 4. 时间序列方法
print("\n4. 时间序列方法...")

# 重采样
print("重采样:")
monthly = ts.resample('M').mean()
print(monthly)

# 移动平均
print("\n移动平均:")
ma7 = ts.rolling(window=7).mean()
print(ma7.head(10))

# 差分
print("\n差分:")
diff = ts.diff()
print(diff.head(10))

# 滞后
print("\n滞后:")
lags = ts.shift(1)
print(lags.head(10))

# 5. 时间序列操作
print("\n5. 时间序列操作...")

# 加法
print("加法:")
ts_add = ts + 10
print(ts_add.head())

# 乘法
print("\n乘法:")
ts_mul = ts * 2
print(ts_mul.head())

# 统计运算
print("\n统计运算:")
print("均值:", ts.mean())
print("标准差:", ts.std())
print("最大值:", ts.max())
print("最小值:", ts.min())

# 6. 时间序列分析
print("\n6. 时间序列分析...")

# 季节性分解
print("季节性分解:")
from statsmodels.tsa.seasonal import seasonal_decompose
try:
    decomposition = seasonal_decompose(ts, model='additive', period=30)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    print("趋势:", trend.head())
    print("季节性:", seasonal.head())
    print("残差:", residual.head())
except ImportError:
    print("statsmodels库未安装，跳过季节性分解")

# 7. 时间序列可视化
print("\n7. 时间序列可视化...")

# 创建可视化数据
fig, axes = plt.subplots(3, 2, figsize=(14, 16))

# 原始时间序列
ax = axes[0, 0]
ts.plot(ax=ax, color='steelblue')
ax.set_title('原始时间序列', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3)

# 月度重采样
ax = axes[0, 1]
monthly.plot(ax=ax, color='coral')
ax.set_title('月度重采样', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3)

# 7天移动平均
ax = axes[1, 0]
ts.plot(ax=ax, color='steelblue', label='原始数据')
ma7.plot(ax=ax, color='coral', label='7天移动平均')
ax.set_title('移动平均', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# 差分
ax = axes[1, 1]
diff.plot(ax=ax, color='green')
ax.set_title('差分', fontsize=12)
ax.set_xlabel('日期', fontsize=10)
ax.set_ylabel('值', fontsize=10)
ax.grid(True, alpha=0.3)

# 季节性分解 (如果statsmodels安装)
try:
    ax = axes[2, 0]
    trend.plot(ax=ax, color='steelblue', label='趋势')
    ax.set_title('趋势', fontsize=12)
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    seasonal.plot(ax=ax, color='coral', label='季节性')
    ax.set_title('季节性', fontsize=12)
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
except (ImportError, NameError):
    # 如果statsmodels未安装，显示其他内容
    ax = axes[2, 0]
    ts.plot(ax=ax, color='steelblue')
    ax.set_title('原始数据', fontsize=12)
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    ts.resample('Q').mean().plot(ax=ax, color='coral')
    ax.set_title('季度重采样', fontsize=12)
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('值', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'time_series_visualization.png'))
print("可视化已保存为 'images/time_series_visualization.png'")

# 8. 性能测试
print("\n8. 性能测试...")
import time

# 测试不同大小时间序列的性能
sizes = [1000, 10000, 100000, 1000000]
times = []

for size in sizes:
    # 创建时间序列
    dates = pd.date_range('2023-01-01', periods=size, freq='H')
    data = np.random.randn(size).cumsum()
ts = pd.Series(data, index=dates)
    
    # 测试重采样性能
    start = time.time()
    ts.resample('D').mean()
    end = time.time()
    times.append(end - start)
    print(f"{size}个时间点重采样耗时: {end - start:.6f}秒")

# 9. 应用示例
print("\n9. 应用示例...")

# 9.1 股票价格分析
print("\n9.1 股票价格分析...")

# 模拟股票价格数据
dates = pd.date_range('2023-01-01', periods=252, freq='B')
price = 100 + np.random.randn(252).cumsum()
stock_data = pd.DataFrame({
    'date': dates,
    'price': price,
    'volume': np.random.randint(100000, 1000000, 252)
})
stock_data.set_index('date', inplace=True)

# 计算收益率
stock_data['return'] = stock_data['price'].pct_change()

# 计算移动平均
stock_data['ma5'] = stock_data['price'].rolling(window=5).mean()
stock_data['ma20'] = stock_data['price'].rolling(window=20).mean()

print("股票数据:")
print(stock_data.head())

# 9.2 销售预测
print("\n9.2 销售预测...")

# 模拟销售数据
dates = pd.date_range('2023-01-01', periods=12, freq='M')
sales = 10000 + np.random.randn(12).cumsum() * 1000
sales_data = pd.DataFrame({
    'date': dates,
    'sales': sales
})
sales_data.set_index('date', inplace=True)

# 计算趋势
sales_data['trend'] = sales_data['sales'].rolling(window=3).mean()

# 预测下一个月销售
last_trend = sales_data['trend'].iloc[-1]
next_month = sales_data.index[-1] + pd.DateOffset(months=1)
print(f"下一个月销售预测: {last_trend:.2f}")

# 10. 总结
print("\n" + "=" * 70)
print("时间序列 (Time Series) 总结")
print("=" * 70)

print("""
Pandas 时间序列功能：

1. 创建时间序列：
   - 从日期范围创建
   - 从DataFrame创建
   - 从CSV/Excel文件创建

2. 时间序列索引：
   - 按日期访问
   - 按月份访问
   - 按年份访问
   - 按日期范围访问

3. 时间序列方法：
   - 重采样：resample()
   - 移动平均：rolling()
   - 差分：diff()
   - 滞后：shift()

4. 时间序列操作：
   - 基本运算
   - 统计运算
   - 逻辑运算

5. 时间序列分析：
   - 季节性分解
   - 趋势分析
   - 周期性分析

6. 应用场景：
   - 股票价格分析
   - 销售预测
   - 天气数据分析
   - 能源消耗分析

7. 性能考虑：
   - 大型时间序列的处理
   - 内存使用优化
   - 计算效率
""")

print("=" * 70)
print("时间序列 (Time Series) 演示完成！")
print("=" * 70)