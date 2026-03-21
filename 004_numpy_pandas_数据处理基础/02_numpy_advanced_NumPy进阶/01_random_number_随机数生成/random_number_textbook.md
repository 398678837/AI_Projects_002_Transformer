# 随机数生成教材

## 第一章：随机数生成基础

### 1.1 什么是随机数

随机数是指在一定范围内随机产生的数字，没有明显的规律。在计算机科学中，随机数通常是通过算法生成的伪随机数，这些数看起来是随机的，但实际上是由确定的算法生成的。

### 1.2 随机数的应用

- **蒙特卡洛模拟**：用于数值积分、风险评估等
- **统计抽样**：从数据集中随机抽取样本
- **密码学**：生成加密密钥和随机密码
- **游戏开发**：生成随机事件和行为
- **机器学习**：随机初始化参数、数据增强
- **实验设计**：随机分配实验组和对照组

### 1.3 NumPy中的随机数生成

NumPy提供了强大的随机数生成功能，支持多种分布的随机数生成，以及各种随机采样和排列功能。

## 第二章：基本随机数生成

### 2.1 均匀分布

均匀分布是最基本的随机分布，所有值的出现概率相等。

```python
import numpy as np

# 生成0-1之间的均匀分布随机数
random_numbers = np.random.rand(5)  # 生成5个随机数
print("均匀分布随机数:", random_numbers)

# 生成多维数组
random_array = np.random.rand(2, 3)  # 生成2x3的数组
print("2x3均匀分布数组:")
print(random_array)
```

### 2.2 正态分布

正态分布（高斯分布）是一种常见的连续概率分布，具有钟形曲线。

```python
# 生成标准正态分布随机数（均值为0，标准差为1）
normal_numbers = np.random.randn(5)  # 生成5个随机数
print("标准正态分布随机数:", normal_numbers)

# 生成多维数组
normal_array = np.random.randn(2, 3)  # 生成2x3的数组
print("2x3正态分布数组:")
print(normal_array)

# 生成指定均值和标准差的正态分布
mean = 5
std = 2
normal_custom = mean + std * np.random.randn(5)
print("均值为5，标准差为2的正态分布:", normal_custom)
```

### 2.3 整数随机数

生成指定范围内的整数随机数。

```python
# 生成指定范围内的整数随机数
integers = np.random.randint(0, 10, size=5)  # 生成0-9之间的5个整数
print("0-9之间的整数:", integers)

# 生成多维数组
integer_array = np.random.randint(1, 101, size=(2, 3))  # 生成1-100之间的2x3数组
print("1-100之间的2x3整数数组:")
print(integer_array)
```

## 第三章：随机数生成器

### 3.1 新的随机数生成器

NumPy 1.17+ 引入了新的随机数生成系统，使用 `default_rng()` 创建随机数生成器。

```python
# 创建随机数生成器
rng = np.random.default_rng()

# 使用生成器生成随机数
print("均匀分布:", rng.random(5))
print("正态分布:", rng.standard_normal(5))
print("整数:", rng.integers(0, 10, size=5))
```

### 3.2 设置种子

设置种子可以保证随机数的可重复性，这在需要复现结果时非常重要。

```python
# 设置种子
rng = np.random.default_rng(42)  # 种子为42

# 生成随机数
print("第一次生成:", rng.random(5))

# 重新创建生成器并使用相同的种子
rng2 = np.random.default_rng(42)
print("第二次生成:", rng2.random(5))  # 结果与第一次相同
```

### 3.3 旧的随机数生成方法

在NumPy 1.17之前，使用的是旧的随机数生成系统，通过全局状态管理随机数。

```python
# 旧方法设置种子
np.random.seed(42)

# 生成随机数
print("均匀分布:", np.random.rand(5))
print("正态分布:", np.random.randn(5))
print("整数:", np.random.randint(0, 10, size=5))
```

## 第四章：不同分布的随机数

### 4.1 二项分布

二项分布用于模拟n次独立的是/非试验，每次成功概率为p。

```python
rng = np.random.default_rng(42)

# 生成二项分布随机数
# n: 试验次数, p: 成功概率, size: 生成数量
binomial = rng.binomial(n=10, p=0.5, size=10)
print("二项分布 (n=10, p=0.5):", binomial)
```

### 4.2 泊松分布

泊松分布用于模拟单位时间内随机事件发生的次数。

```python
# 生成泊松分布随机数
# lam: 事件发生率, size: 生成数量
poisson = rng.poisson(lam=3, size=10)
print("泊松分布 (lambda=3):", poisson)
```

### 4.3 指数分布

指数分布用于模拟事件发生的时间间隔。

```python
# 生成指数分布随机数
# scale: 均值, size: 生成数量
exponential = rng.exponential(scale=1, size=10)
print("指数分布 (scale=1):", exponential)
```

### 4.4 伽马分布

伽马分布是一种连续概率分布，常用于模拟等待时间。

```python
# 生成伽马分布随机数
# shape: 形状参数, scale: 尺度参数, size: 生成数量
gamma = rng.gamma(shape=2, scale=1, size=10)
print("伽马分布 (shape=2, scale=1):", gamma)
```

### 4.5 其他分布

NumPy还支持许多其他分布：

- **beta**：贝塔分布
- **chisquare**：卡方分布
- **f**：F分布
- **gamma**：伽马分布
- **geometric**：几何分布
- **laplace**：拉普拉斯分布
- **lognormal**：对数正态分布
- **negative_binomial**：负二项分布
- **normal**：正态分布
- **pareto**：帕累托分布
- **uniform**：均匀分布
- **weibull**：威布尔分布

## 第五章：随机采样

### 5.1 从数组中采样

随机采样是从现有数据集中抽取样本的过程。

```python
rng = np.random.default_rng(42)

# 从数组中随机采样
arr = np.array([1, 2, 3, 4, 5])

# 有放回采样
sample_with_replacement = rng.choice(arr, size=3, replace=True)
print("有放回采样:", sample_with_replacement)

# 无放回采样
sample_without_replacement = rng.choice(arr, size=3, replace=False)
print("无放回采样:", sample_without_replacement)
```

### 5.2 加权采样

加权采样允许根据权重分配不同的采样概率。

```python
# 加权采样
weights = [0.1, 0.2, 0.3, 0.2, 0.2]  # 权重
weighted_sample = rng.choice(arr, size=5, p=weights)
print("加权采样:", weighted_sample)

# 验证权重
unique, counts = np.unique(weighted_sample, return_counts=True)
print("采样结果计数:", dict(zip(unique, counts)))
```

## 第六章：随机排列

### 6.1 随机打乱数组

随机打乱数组可以用于随机化数据顺序。

```python
rng = np.random.default_rng(42)

# 随机打乱数组（原地操作）
arr = np.array([1, 2, 3, 4, 5])
rng.shuffle(arr)
print("打乱后数组:", arr)
```

### 6.2 生成随机排列

生成随机排列可以用于创建随机索引。

```python
# 生成随机排列
permutation = rng.permutation(10)  # 生成0-9的随机排列
print("0-9的随机排列:", permutation)

# 对现有数组生成随机排列
arr = np.array([1, 2, 3, 4, 5])
permuted = rng.permutation(arr)
print("数组的随机排列:", permuted)
print("原始数组:", arr)  # 原始数组不变
```

## 第七章：性能考虑

### 7.1 生成大型数组

生成大型随机数数组时需要考虑性能。

```python
import time

# 测试生成大型数组的性能
rng = np.random.default_rng()

sizes = [1000, 10000, 100000, 1000000]
for size in sizes:
    start = time.time()
    data = rng.random(size)
    end = time.time()
    print(f"生成{size}个随机数耗时: {end - start:.6f}秒")
```

### 7.2 不同分布的性能

不同分布的随机数生成性能可能不同。

```python
# 测试不同分布的性能
distributions = ['uniform', 'normal', 'binomial', 'poisson']
size = 1000000

for dist in distributions:
    start = time.time()
    if dist == 'uniform':
        rng.random(size)
    elif dist == 'normal':
        rng.standard_normal(size)
    elif dist == 'binomial':
        rng.binomial(n=10, p=0.5, size=size)
    elif dist == 'poisson':
        rng.poisson(lam=3, size=size)
    end = time.time()
    print(f"{dist}分布耗时: {end - start:.6f}秒")
```

## 第八章：应用示例

### 8.1 蒙特卡洛方法计算π

蒙特卡洛方法是一种通过随机采样来解决问题的数值方法。

```python
def estimate_pi(n_points):
    rng = np.random.default_rng()
    x = rng.random(n_points)
    y = rng.random(n_points)
    distance = np.sqrt(x**2 + y**2)
    inside_circle = distance <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_points
    return pi_estimate

# 估计π值
n_points = 1000000
pi_estimate = estimate_pi(n_points)
print(f"π的估计值: {pi_estimate}")
print(f"误差: {abs(pi_estimate - np.pi):.6f}")
```

### 8.2 模拟股票价格

使用随机数模拟股票价格的变化。

```python
def simulate_stock_price(initial_price, days, mu, sigma):
    rng = np.random.default_rng()
    # 生成每日收益率
    returns = rng.normal(loc=mu/252, scale=sigma/np.sqrt(252), size=days)
    # 计算价格
    prices = initial_price * np.exp(np.cumsum(returns))
    return prices

# 模拟股票价格
initial_price = 100
days = 252
mu = 0.08  # 年收益率
sigma = 0.2  # 年波动率

prices = simulate_stock_price(initial_price, days, mu, sigma)
print(f"初始价格: {initial_price}")
print(f"最终价格: {prices[-1]:.2f}")
print(f"收益率: {(prices[-1]/initial_price - 1)*100:.2f}%")
```

### 8.3 生成随机密码

使用随机数生成安全的密码。

```python
def generate_password(length=12):
    import string
    rng = np.random.default_rng()
    characters = string.ascii_letters + string.digits + string.punctuation
    password = ''.join(rng.choice(list(characters), size=length))
    return password

# 生成随机密码
password = generate_password(16)
print(f"随机密码: {password}")
```

### 8.4 随机抽样

从数据集中随机抽取样本。

```python
def random_sample(data, sample_size):
    rng = np.random.default_rng()
    if sample_size > len(data):
        raise ValueError("样本大小不能大于数据大小")
    indices = rng.choice(len(data), size=sample_size, replace=False)
    return data[indices]

# 测试随机抽样
data = np.arange(100)
sample = random_sample(data, 10)
print("随机样本:", sample)
```

## 第九章：最佳实践

### 9.1 使用新的随机数生成器

推荐使用新的随机数生成器 `default_rng()`，它提供了更好的性能和可重复性。

```python
# 推荐使用新的随机数生成器
rng = np.random.default_rng()

# 不推荐使用旧的全局状态
# np.random.seed(42)  # 旧方法
```

### 9.2 设置种子的重要性

在需要复现结果的情况下，设置种子是非常重要的。

```python
# 为了可重复性，设置种子
rng = np.random.default_rng(42)  # 固定种子

# 生成随机数
print(rng.random(5))  # 每次运行结果相同
```

### 9.3 性能优化

生成大型随机数数组时，可以通过预分配内存来提高性能。

```python
# 预分配内存
size = 1000000
data = np.empty(size)

# 一次性生成随机数
rng = np.random.default_rng()
data[:] = rng.random(size)
```

### 9.4 安全性考虑

注意：NumPy的随机数生成器不适合密码学应用，对于密码学应用，应使用专门的密码学随机数生成器。

```python
# 注意：NumPy的随机数生成器不适合密码学应用
# 对于密码学应用，使用专门的密码学随机数生成器
import secrets

# 生成安全的随机密码
secure_password = ''.join(secrets.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(16))
print(f"安全密码: {secure_password}")
```

## 第十章：习题

### 10.1 选择题

1. 以下哪个函数用于生成0-1之间的均匀分布随机数？
   - A) np.random.randn()
   - B) np.random.rand()
   - C) np.random.randint()
   - D) np.random.normal()

2. 以下哪个函数用于设置随机数种子？
   - A) np.random.seed()
   - B) np.random.default_rng()
   - C) np.random.shuffle()
   - D) np.random.permutation()

3. 以下哪个分布用于模拟单位时间内随机事件发生的次数？
   - A) 二项分布
   - B) 正态分布
   - C) 泊松分布
   - D) 指数分布

### 10.2 填空题

1. NumPy 1.17+ 引入的新随机数生成器是________________。
2. 生成指定均值和标准差的正态分布随机数的公式是________________。
3. 无放回采样的参数设置是________________。

### 10.3 简答题

1. 简述伪随机数和真随机数的区别。
2. 简述设置种子的重要性。
3. 简述蒙特卡洛方法的基本原理。

### 10.4 编程题

1. 使用蒙特卡洛方法估计圆周率π，要求估计值与真实值的误差小于0.001。
2. 生成一个10x10的随机矩阵，其中元素为0-9的整数。
3. 从数组[1, 2, 3, 4, 5]中进行加权采样，权重分别为[0.1, 0.2, 0.3, 0.2, 0.2]，生成100个样本并统计每个元素的出现次数。
4. 模拟抛硬币1000次，统计正面和反面的次数。

## 第十一章：总结

### 11.1 知识回顾

1. **基本随机数生成**：均匀分布、正态分布、整数随机数
2. **随机数生成器**：新的default_rng()和旧的seed()方法
3. **不同分布**：二项分布、泊松分布、指数分布等
4. **随机采样**：有放回和无放回采样，加权采样
5. **随机排列**：打乱数组，生成随机排列
6. **应用场景**：蒙特卡洛模拟、股票价格模拟、密码生成
7. **最佳实践**：使用新的生成器、设置种子、性能优化

### 11.2 学习建议

1. **实践练习**：多练习不同分布的随机数生成
2. **性能测试**：测试不同方法的性能
3. **应用开发**：在实际项目中应用随机数生成
4. **理论学习**：了解随机数生成的数学原理
5. **安全性**：了解密码学安全的随机数生成

### 11.3 进阶学习

1. 随机数生成的数学原理
2. 其他随机数生成库（如random模块）
3. 密码学安全的随机数生成
4. 并行随机数生成
5. 随机数生成在机器学习中的应用