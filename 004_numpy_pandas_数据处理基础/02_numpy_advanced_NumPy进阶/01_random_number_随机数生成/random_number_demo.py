"""
随机数生成演示
Random Number Generation Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("随机数生成 (Random Number Generation) 演示")
print("=" * 70)

# 1. 基本随机数生成
print("\n1. 基本随机数生成...")

# 均匀分布
print("均匀分布:")
uniform = np.random.rand(5)
print("0-1之间的随机数:", uniform)

# 正态分布
print("\n正态分布:")
normal = np.random.randn(5)
print("标准正态分布随机数:", normal)

# 整数随机数
print("\n整数随机数:")
integers = np.random.randint(0, 10, size=5)
print("0-9之间的整数:", integers)

# 2. 随机数生成器
print("\n2. 随机数生成器...")

# 创建随机数生成器
rng = np.random.default_rng(42)  # 设置种子

# 使用生成器生成随机数
print("使用生成器生成随机数:")
print("均匀分布:", rng.random(5))
print("正态分布:", rng.standard_normal(5))
print("整数:", rng.integers(0, 10, size=5))

# 3. 不同分布的随机数
print("\n3. 不同分布的随机数...")

# 二项分布
print("二项分布:")
binomial = rng.binomial(n=10, p=0.5, size=10)
print("n=10, p=0.5:", binomial)

# 泊松分布
print("\n泊松分布:")
poisson = rng.poisson(lam=3, size=10)
print("lambda=3:", poisson)

# 指数分布
print("\n指数分布:")
exponential = rng.exponential(scale=1, size=10)
print("scale=1:", exponential)

# 伽马分布
print("\n伽马分布:")
gamma = rng.gamma(shape=2, scale=1, size=10)
print("shape=2, scale=1:", gamma)

# 4. 随机采样
print("\n4. 随机采样...")

# 从数组中随机采样
print("从数组中随机采样:")
arr = np.array([1, 2, 3, 4, 5])
sample = rng.choice(arr, size=3, replace=True)
print("有放回采样:", sample)
sample_no_replace = rng.choice(arr, size=3, replace=False)
print("无放回采样:", sample_no_replace)

# 加权采样
print("\n加权采样:")
weights = [0.1, 0.2, 0.3, 0.2, 0.2]
sample_weighted = rng.choice(arr, size=5, p=weights)
print("加权采样:", sample_weighted)

# 5. 随机排列
print("\n5. 随机排列...")

# 随机打乱数组
print("随机打乱数组:")
arr = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr)
print("打乱后:", arr)

# 生成随机排列
print("\n生成随机排列:")
permutation = rng.permutation(10)
print("0-9的随机排列:", permutation)

# 6. 性能测试
print("\n6. 性能测试...")
import time

# 测试不同方法的性能
sizes = [1000, 10000, 100000, 1000000]
methods = ['rand', 'randn', 'integers']
times = {method: [] for method in methods}

for size in sizes:
    for method in methods:
        start = time.time()
        if method == 'rand':
            np.random.rand(size)
        elif method == 'randn':
            np.random.randn(size)
        elif method == 'integers':
            np.random.randint(0, 100, size=size)
        end = time.time()
        times[method].append(end - start)
        print(f"{method}({size}): {end - start:.6f}秒")

# 7. 可视化
print("\n7. 可视化...")

# 创建可视化数据
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 均匀分布
ax = axes[0, 0]
data = rng.random(10000)
ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue')
ax.set_title('均匀分布 (Uniform Distribution)', fontsize=12)
ax.set_xlabel('值', fontsize=10)
ax.set_ylabel('概率密度', fontsize=10)
ax.grid(True, alpha=0.3)

# 正态分布
ax = axes[0, 1]
data = rng.standard_normal(10000)
ax.hist(data, bins=50, density=True, alpha=0.7, color='coral')
ax.set_title('正态分布 (Normal Distribution)', fontsize=12)
ax.set_xlabel('值', fontsize=10)
ax.set_ylabel('概率密度', fontsize=10)
ax.grid(True, alpha=0.3)

# 二项分布
ax = axes[1, 0]
data = rng.binomial(n=10, p=0.5, size=10000)
ax.hist(data, bins=11, density=True, alpha=0.7, color='green')
ax.set_title('二项分布 (Binomial Distribution)', fontsize=12)
ax.set_xlabel('成功次数', fontsize=10)
ax.set_ylabel('概率', fontsize=10)
ax.grid(True, alpha=0.3)

# 泊松分布
ax = axes[1, 1]
data = rng.poisson(lam=3, size=10000)
ax.hist(data, bins=20, density=True, alpha=0.7, color='purple')
ax.set_title('泊松分布 (Poisson Distribution)', fontsize=12)
ax.set_xlabel('事件数', fontsize=10)
ax.set_ylabel('概率', fontsize=10)
ax.grid(True, alpha=0.3)

# 指数分布
ax = axes[2, 0]
data = rng.exponential(scale=1, size=10000)
ax.hist(data, bins=50, density=True, alpha=0.7, color='orange')
ax.set_title('指数分布 (Exponential Distribution)', fontsize=12)
ax.set_xlabel('值', fontsize=10)
ax.set_ylabel('概率密度', fontsize=10)
ax.grid(True, alpha=0.3)

# 性能对比
ax = axes[2, 1]
for method in methods:
    ax.plot(sizes, times[method], marker='o', label=method)
ax.set_title('随机数生成性能对比', fontsize=12)
ax.set_xlabel('数组大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'random_numbers.png'))
print("可视化已保存为 'images/random_numbers.png'")

# 8. 应用示例
print("\n8. 应用示例...")

# 模拟硬币翻转
print("模拟硬币翻转:")
flips = rng.binomial(n=1, p=0.5, size=1000)
heads = np.sum(flips)
tails = 1000 - heads
print(f"正面: {heads}, 反面: {tails}")
print(f"正面概率: {heads/1000:.3f}")

# 生成随机密码
print("\n生成随机密码:")
import string
characters = string.ascii_letters + string.digits + string.punctuation
password = ''.join(rng.choice(list(characters), size=12))
print(f"随机密码: {password}")

# 蒙特卡洛方法计算π
print("\n蒙特卡洛方法计算π:")
n_points = 1000000
x = rng.random(n_points)
y = rng.random(n_points)
distance = np.sqrt(x**2 + y**2)
inside_circle = distance <= 1
pi_estimate = 4 * np.sum(inside_circle) / n_points
print(f"π的估计值: {pi_estimate}")
print(f"误差: {abs(pi_estimate - np.pi):.6f}")

# 9. 总结
print("\n" + "=" * 70)
print("随机数生成总结")
print("=" * 70)

print("""
NumPy随机数生成功能：

1. 基本随机数：
   - np.random.rand()：0-1均匀分布
   - np.random.randn()：标准正态分布
   - np.random.randint()：整数随机数

2. 随机数生成器：
   - np.random.default_rng()：新的随机数生成器
   - 支持设置种子，保证可重复性

3. 分布类型：
   - 均匀分布、正态分布、二项分布
   - 泊松分布、指数分布、伽马分布
   - 以及更多其他分布

4. 随机采样：
   - 有放回采样和无放回采样
   - 加权采样

5. 随机排列：
   - np.random.shuffle()：原地打乱
   - np.random.permutation()：返回新数组

6. 应用场景：
   - 蒙特卡洛模拟
   - 随机抽样
   - 密码生成
   - 游戏开发
   - 统计测试

7. 性能考虑：
   - 生成大型数组时的性能
   - 不同分布的计算开销
   - 种子设置的重要性
""")

print("=" * 70)
print("随机数生成演示完成！")
print("=" * 70)