# 文件I/O教材

## 第一章：文件I/O基础

### 1.1 什么是文件I/O

文件I/O（Input/Output）是指与文件系统进行交互的操作，包括读取和写入文件。在NumPy中，文件I/O功能非常强大，支持多种文件格式和操作方式。

### 1.2 文件I/O的重要性

- **数据持久化**：将内存中的数据保存到磁盘
- **数据交换**：与其他程序或系统交换数据
- **模型保存**：保存训练好的模型参数
- **配置管理**：读取和写入配置文件
- **大型数据处理**：处理超出内存大小的数据集

### 1.3 NumPy文件I/O的特点

- **多种文件格式**：支持.npy, .npz, .txt, .csv等格式
- **高效存储**：二进制格式保存，节省空间和时间
- **类型安全**：保留数据类型信息
- **灵活性**：支持结构化数据和大型数据集

## 第二章：NumPy文件格式

### 2.1 .npy格式

.npy是NumPy的专用二进制格式，用于存储单个NumPy数组。它保存了数组的数据、形状、数据类型等信息。

```python
import numpy as np
import os

# 创建示例数据
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 保存为.npy文件
npy_file = 'array.npy'
np.save(npy_file, arr)
print(f"数组已保存到 {npy_file}")

# 加载.npy文件
loaded_arr = np.load(npy_file)
print("加载的数组:")
print(loaded_arr)
```

### 2.2 .npz格式

.npz是NumPy的压缩文件格式，用于存储多个NumPy数组。它实际上是一个ZIP文件，包含多个.npy文件。

```python
# 创建多个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# 保存为.npz文件
npz_file = 'arrays.npz'
np.savez(npz_file, arr1=arr1, arr2=arr2, arr3=arr3)
print(f"多个数组已保存到 {npz_file}")

# 加载.npz文件
loaded_data = np.load(npz_file)
print("加载的数组:")
print("数组1:", loaded_data['arr1'])
print("数组2:", loaded_data['arr2'])
print("数组3:", loaded_data['arr3'])
```

### 2.3 文本文件格式

NumPy支持读取和写入文本文件，如.txt和.csv文件。

```python
# 创建示例数据
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 保存为文本文件
txt_file = 'array.txt'
np.savetxt(txt_file, arr, fmt='%d', delimiter=',')
print(f"数组已保存到 {txt_file}")

# 加载文本文件
loaded_txt = np.loadtxt(txt_file, delimiter=',')
print("加载的数组:")
print(loaded_txt)
```

### 2.4 二进制文件格式

NumPy也支持直接读写原始二进制文件。

```python
# 保存为二进制文件
bin_file = 'array.bin'
arr.tofile(bin_file)
print(f"数组已保存到 {bin_file}")

# 加载二进制文件
loaded_bin = np.fromfile(bin_file, dtype=np.int64)
loaded_bin = loaded_bin.reshape((3, 3))
print("加载的数组:")
print(loaded_bin)
```

## 第三章：核心函数

### 3.1 保存和加载单个数组

- **np.save(file, arr)**：将数组保存为.npy文件
- **np.load(file)**：加载.npy文件

```python
# 保存数组
np.save('data.npy', arr)

# 加载数组
loaded = np.load('data.npy')
```

### 3.2 保存和加载多个数组

- **np.savez(file, *args, **kwds)**：将多个数组保存为.npz文件
- **np.savez_compressed(file, *args, **kwds)**：将多个数组保存为压缩的.npz文件

```python
# 保存多个数组
np.savez('data.npz', arr1=arr1, arr2=arr2)

# 加载多个数组
loaded = np.load('data.npz')
arr1 = loaded['arr1']
arr2 = loaded['arr2']
```

### 3.3 文本文件I/O

- **np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')**：将数组保存为文本文件
- **np.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes')**：从文本文件加载数组

```python
# 保存为文本文件
np.savetxt('data.txt', arr, fmt='%d', delimiter=',')

# 加载文本文件
loaded = np.loadtxt('data.txt', delimiter=',')
```

### 3.4 二进制文件I/O

- **ndarray.tofile(fid, sep='', format='%s')**：将数组保存为二进制文件
- **np.fromfile(file, dtype=float, count=-1, sep='', offset=0)**：从二进制文件加载数组

```python
# 保存为二进制文件
arr.tofile('data.bin')

# 加载二进制文件
loaded = np.fromfile('data.bin', dtype=np.float64)
```

### 3.5 内存映射文件

- **np.memmap(filename, dtype=float, mode='r+', offset=0, shape=None, order='C')**：创建内存映射文件

```python
# 创建内存映射文件
fp = np.memmap('large_array.dat', dtype='float64', mode='w+', shape=(1000, 1000))
fp[:] = np.random.rand(1000, 1000)[:]
del fp  # 关闭文件

# 加载内存映射文件
fp = np.memmap('large_array.dat', dtype='float64', mode='r', shape=(1000, 1000))
print(fp.shape)
del fp  # 关闭文件
```

## 第四章：结构化数组I/O

### 4.1 保存和加载结构化数组

结构化数组是一种包含不同数据类型的数组，可以保存为.npy文件。

```python
# 创建结构化数组
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f8')]
data = [('Alice', 25, 1.65), ('Bob', 30, 1.80), ('Charlie', 35, 1.75)]
structured_arr = np.array(data, dtype=dtype)

# 保存结构化数组
np.save('structured.npy', structured_arr)

# 加载结构化数组
loaded_struct = np.load('structured.npy')
print(loaded_struct)
```

## 第五章：性能考虑

### 5.1 文件格式比较

不同文件格式的读写性能和文件大小有所不同：

| 格式 | 读写速度 | 文件大小 | 特点 |
|------|----------|----------|------|
| .npy | 快 | 小 | 二进制格式，保留数据类型 |
| .npz | 较快 | 最小 | 压缩格式，适合多个数组 |
| .txt | 慢 | 大 | 人类可读，跨平台 |
| .bin | 快 | 小 | 原始二进制，需要指定数据类型 |

### 5.2 性能测试

```python
import time

# 测试不同文件格式的读写性能
size = 1000
arr = np.random.rand(size, size)

# 测试.npy格式
start = time.time()
np.save('test.npy', arr)
write_time = time.time() - start

start = time.time()
loaded = np.load('test.npy')
read_time = time.time() - start

print(f".npy: 写入 {write_time:.6f}s, 读取 {read_time:.6f}s")

# 测试.txt格式
start = time.time()
np.savetxt('test.txt', arr)
write_time = time.time() - start

start = time.time()
loaded = np.loadtxt('test.txt')
read_time = time.time() - start

print(f".txt: 写入 {write_time:.6f}s, 读取 {read_time:.6f}s")
```

### 5.3 内存使用

- **内存映射文件**：适合处理大型数组，不需要将整个数组加载到内存
- **分块读写**：对于大型数据集，可以分块读写以减少内存使用
- **数据压缩**：使用.npz格式可以减少文件大小

## 第六章：应用场景

### 6.1 数据持久化

```python
# 生成模拟数据
data = np.random.rand(1000, 100)  # 1000个样本，100个特征
labels = np.random.randint(0, 2, size=1000)  # 二分类标签

# 保存数据
np.savez('dataset.npz', data=data, labels=labels)
print("数据集已保存")

# 加载数据
loaded_dataset = np.load('dataset.npz')
loaded_data = loaded_dataset['data']
loaded_labels = loaded_dataset['labels']
print(f"加载的数据集: 数据形状 {loaded_data.shape}, 标签形状 {loaded_labels.shape}")
```

### 6.2 模型参数保存

```python
# 模拟模型参数
weights = np.random.rand(100, 10)  # 权重
biases = np.random.rand(10)  # 偏置

# 保存模型参数
np.savez('model_params.npz', weights=weights, biases=biases)
print("模型参数已保存")

# 加载模型参数
loaded_params = np.load('model_params.npz')
loaded_weights = loaded_params['weights']
loaded_biases = loaded_params['biases']
print(f"加载的模型参数: 权重形状 {loaded_weights.shape}, 偏置形状 {loaded_biases.shape}")
```

### 6.3 配置文件读写

```python
# 创建配置数组
config = np.array([
    [0.001, 1000, 32],  # 学习率, 迭代次数, 批次大小
    [0.01, 500, 64],
    [0.1, 100, 128]
])

# 保存配置
np.savetxt('config.txt', config, fmt='%.3f', delimiter=',', 
           header='learning_rate,epochs,batch_size')
print("配置已保存")

# 加载配置
loaded_config = np.loadtxt('config.txt', delimiter=',', skiprows=1)
print("加载的配置:")
print(loaded_config)
```

### 6.4 大型数据处理

```python
# 创建大型数组
size = 10000
total_size = size * size

# 使用内存映射文件
fp = np.memmap('large_array.dat', dtype='float64', mode='w+', shape=(size, size))

# 分块写入数据
block_size = 1000
for i in range(0, size, block_size):
    end = min(i + block_size, size)
    fp[i:end, :] = np.random.rand(end - i, size)
    print(f"已写入 {end} / {size} 行")

del fp  # 关闭文件
print("大型数组已保存")

# 分块读取数据
fp = np.memmap('large_array.dat', dtype='float64', mode='r', shape=(size, size))
print(f"数组形状: {fp.shape}")
print(f"前10行前10列: {fp[:10, :10]}")
del fp  # 关闭文件
```

## 第七章：最佳实践

### 7.1 文件格式选择

- **.npy**：适合单个NumPy数组，保存数据类型和形状信息
- **.npz**：适合多个数组，支持压缩
- **.txt/.csv**：适合人类可读的数据，跨平台兼容
- **二进制文件**：适合原始数据，需要指定数据类型
- **内存映射文件**：适合大型数据集

### 7.2 性能优化

1. **批量读写**：避免频繁的文件I/O操作
2. **选择合适的文件格式**：根据数据大小和使用场景选择
3. **使用压缩**：对于大型数据集，使用.npz格式
4. **内存映射**：对于超大数据集，使用内存映射文件
5. **数据类型优化**：选择合适的数据类型以减少文件大小

### 7.3 数据安全

1. **备份重要数据**：定期备份数据文件
2. **错误处理**：使用try-except处理文件I/O错误
3. **数据验证**：加载数据后验证数据的完整性
4. **文件权限**：设置适当的文件权限

### 7.4 代码组织

1. **封装I/O函数**：将文件I/O操作封装为函数
2. **统一路径管理**：使用配置文件或环境变量管理文件路径
3. **使用上下文管理器**：对于文件操作，使用with语句
4. **日志记录**：记录文件I/O操作的状态

## 第八章：常见问题和解决方案

### 8.1 文件路径问题

**问题**：文件路径不正确导致文件找不到

**解决方案**：使用绝对路径或相对于当前工作目录的路径

```python
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

# 确保目录存在
os.makedirs(data_dir, exist_ok=True)

# 构建文件路径
file_path = os.path.join(data_dir, 'array.npy')
```

### 8.2 数据类型不匹配

**问题**：加载二进制文件时数据类型不匹配

**解决方案**：确保加载时使用正确的数据类型

```python
# 保存时的数据类型
arr = np.array([1, 2, 3], dtype=np.int32)
arr.tofile('data.bin')

# 加载时使用相同的数据类型
loaded = np.fromfile('data.bin', dtype=np.int32)
```

### 8.3 内存不足

**问题**：加载大型数组时内存不足

**解决方案**：使用内存映射文件或分块加载

```python
# 使用内存映射文件
fp = np.memmap('large_array.dat', dtype='float64', mode='r', shape=(10000, 10000))

# 分块处理
for i in range(0, 10000, 1000):
    block = fp[i:i+1000, :]
    # 处理块数据
```

### 8.4 文件格式不兼容

**问题**：不同版本的NumPy可能有不同的文件格式

**解决方案**：使用较新的NumPy版本或明确指定格式

```python
# 保存时指定格式
np.save('array.npy', arr)

# 加载时使用try-except处理可能的格式问题
try:
    loaded = np.load('array.npy')
except Exception as e:
    print(f"加载错误: {e}")
```

## 第九章：习题

### 9.1 选择题

1. 以下哪个文件格式是NumPy的专用二进制格式？
   - A) .txt
   - B) .csv
   - C) .npy
   - D) .bin

2. 以下哪个函数用于保存多个NumPy数组？
   - A) np.save()
   - B) np.savez()
   - C) np.savetxt()
   - D) np.tofile()

3. 以下哪个函数用于创建内存映射文件？
   - A) np.load()
   - B) np.memmap()
   - C) np.fromfile()
   - D) np.loadtxt()

### 9.2 填空题

1. NumPy的压缩文件格式是________________。
2. 保存数组为文本文件的函数是________________。
3. 内存映射文件的主要优势是________________。

### 9.3 简答题

1. 简述.npy和.npz文件格式的区别。
2. 简述内存映射文件的工作原理。
3. 简述如何选择合适的文件格式。

### 9.4 编程题

1. 创建一个5x5的随机矩阵，分别保存为.npy、.npz、.txt和.bin文件，然后加载并验证数据的正确性。
2. 生成一个1000x1000的大型随机矩阵，使用内存映射文件保存和加载。
3. 创建一个结构化数组，包含姓名、年龄和身高信息，保存为.npy文件并加载。
4. 编写一个函数，将NumPy数组保存为CSV文件，并处理文件路径不存在的情况。

## 第十章：总结

### 10.1 知识回顾

1. **文件格式**：.npy, .npz, .txt, .csv, 二进制文件
2. **核心函数**：np.save(), np.load(), np.savez(), np.savetxt(), np.loadtxt(), np.memmap()
3. **性能优化**：文件格式选择、内存映射、分块处理
4. **应用场景**：数据持久化、模型保存、配置管理、大型数据处理
5. **最佳实践**：文件格式选择、性能优化、数据安全、代码组织

### 10.2 学习建议

1. **实践练习**：多练习不同文件格式的读写操作
2. **性能测试**：测试不同文件格式的性能
3. **应用开发**：在实际项目中应用文件I/O操作
4. **代码优化**：学习如何优化文件I/O性能
5. **安全性**：学习如何保证数据的安全性

### 10.3 进阶学习

1. 文件格式的内部结构
2. 并行文件I/O
3. 分布式文件系统
4. 数据压缩算法
5. 数据序列化