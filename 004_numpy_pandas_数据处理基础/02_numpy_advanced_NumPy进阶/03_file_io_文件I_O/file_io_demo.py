"""
文件I/O演示
File I/O Demo
"""

import numpy as np
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

print("=" * 70)
print("文件I/O (File I/O) 演示")
print("=" * 70)

# 确保数据目录存在
os.makedirs(data_dir, exist_ok=True)

# 1. 保存和加载NumPy数组
print("\n1. 保存和加载NumPy数组...")

# 创建示例数据
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原始数组:")
print(arr)

# 保存为.npy文件
npy_file = os.path.join(data_dir, 'array.npy')
np.save(npy_file, arr)
print(f"\n数组已保存到 {npy_file}")

# 加载.npy文件
loaded_arr = np.load(npy_file)
print("\n加载的数组:")
print(loaded_arr)

# 2. 保存和加载多个数组
print("\n2. 保存和加载多个数组...")

# 创建多个数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

print("数组1:", arr1)
print("数组2:", arr2)
print("数组3:", arr3)

# 保存为.npz文件
npz_file = os.path.join(data_dir, 'arrays.npz')
np.savez(npz_file, arr1=arr1, arr2=arr2, arr3=arr3)
print(f"\n多个数组已保存到 {npz_file}")

# 加载.npz文件
loaded_data = np.load(npz_file)
print("\n加载的数组:")
print("数组1:", loaded_data['arr1'])
print("数组2:", loaded_data['arr2'])
print("数组3:", loaded_data['arr3'])

# 3. 文本文件I/O
print("\n3. 文本文件I/O...")

# 创建示例数据
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 保存为文本文件
txt_file = os.path.join(data_dir, 'array.txt')
np.savetxt(txt_file, arr, fmt='%d', delimiter=',')
print(f"\n数组已保存到 {txt_file}")

# 加载文本文件
loaded_txt = np.loadtxt(txt_file, delimiter=',')
print("\n加载的数组:")
print(loaded_txt)

# 4. CSV文件I/O
print("\n4. CSV文件I/O...")

# 保存为CSV文件
csv_file = os.path.join(data_dir, 'array.csv')
np.savetxt(csv_file, arr, fmt='%d', delimiter=',')
print(f"\n数组已保存到 {csv_file}")

# 加载CSV文件
loaded_csv = np.loadtxt(csv_file, delimiter=',')
print("\n加载的数组:")
print(loaded_csv)

# 5. 二进制文件I/O
print("\n5. 二进制文件I/O...")

# 保存为二进制文件
bin_file = os.path.join(data_dir, 'array.bin')
arr.tofile(bin_file)
print(f"\n数组已保存到 {bin_file}")

# 加载二进制文件
loaded_bin = np.fromfile(bin_file, dtype=np.int64)
loaded_bin = loaded_bin.reshape((3, 3))
print("\n加载的数组:")
print(loaded_bin)

# 6. 结构化数组I/O
print("\n6. 结构化数组I/O...")

# 创建结构化数组
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f8')]
data = [('Alice', 25, 1.65), ('Bob', 30, 1.80), ('Charlie', 35, 1.75)]
structured_arr = np.array(data, dtype=dtype)

print("原始结构化数组:")
print(structured_arr)

# 保存结构化数组
struct_file = os.path.join(data_dir, 'structured.npy')
np.save(struct_file, structured_arr)
print(f"\n结构化数组已保存到 {struct_file}")

# 加载结构化数组
loaded_struct = np.load(struct_file)
print("\n加载的结构化数组:")
print(loaded_struct)

# 7. 内存映射文件
print("\n7. 内存映射文件...")

# 创建大型数组
large_arr = np.random.rand(1000, 1000)

# 保存为内存映射文件
mmap_file = os.path.join(data_dir, 'large_array.dat')
fp = np.memmap(mmap_file, dtype='float64', mode='w+', shape=large_arr.shape)
fp[:] = large_arr[:]
del fp  # 关闭文件

print(f"\n大型数组已保存到 {mmap_file}")

# 加载内存映射文件
fp = np.memmap(mmap_file, dtype='float64', mode='r', shape=large_arr.shape)
print("\n加载的大型数组形状:", fp.shape)
print("前5x5元素:")
print(fp[:5, :5])
del fp  # 关闭文件

# 8. 性能测试
print("\n8. 性能测试...")
import time

# 测试不同文件格式的读写性能
sizes = [100, 500, 1000, 2000]
formats = ['npy', 'npz', 'txt', 'bin']

for size in sizes:
    print(f"\n测试 {size}x{size} 数组:")
    arr = np.random.rand(size, size)
    
    for fmt in formats:
        # 写入测试
        start = time.time()
        if fmt == 'npy':
            file_path = os.path.join(data_dir, f'test_{size}.npy')
            np.save(file_path, arr)
        elif fmt == 'npz':
            file_path = os.path.join(data_dir, f'test_{size}.npz')
            np.savez(file_path, arr=arr)
        elif fmt == 'txt':
            file_path = os.path.join(data_dir, f'test_{size}.txt')
            np.savetxt(file_path, arr)
        elif fmt == 'bin':
            file_path = os.path.join(data_dir, f'test_{size}.bin')
            arr.tofile(file_path)
        write_time = time.time() - start
        
        # 读取测试
        start = time.time()
        if fmt == 'npy':
            loaded = np.load(file_path)
        elif fmt == 'npz':
            loaded = np.load(file_path)['arr']
        elif fmt == 'txt':
            loaded = np.loadtxt(file_path)
        elif fmt == 'bin':
            loaded = np.fromfile(file_path, dtype='float64')
            loaded = loaded.reshape((size, size))
        read_time = time.time() - start
        
        # 文件大小
        file_size = os.path.getsize(file_path) / 1024  # KB
        
        print(f"  {fmt}: 写入 {write_time:.6f}s, 读取 {read_time:.6f}s, 大小 {file_size:.2f}KB")

# 9. 应用示例
print("\n9. 应用示例...")

# 9.1 数据保存和加载
print("\n9.1 数据保存和加载...")

# 生成模拟数据
data = np.random.rand(1000, 100)  # 1000个样本，100个特征
labels = np.random.randint(0, 2, size=1000)  # 二分类标签

# 保存数据
np.savez(os.path.join(data_dir, 'dataset.npz'), data=data, labels=labels)
print("数据集已保存")

# 加载数据
loaded_dataset = np.load(os.path.join(data_dir, 'dataset.npz'))
loaded_data = loaded_dataset['data']
loaded_labels = loaded_dataset['labels']
print(f"加载的数据集: 数据形状 {loaded_data.shape}, 标签形状 {loaded_labels.shape}")

# 9.2 配置文件读写
print("\n9.2 配置文件读写...")

# 创建配置数组
config = np.array([
    [0.001, 1000, 32],  # 学习率, 迭代次数, 批次大小
    [0.01, 500, 64],
    [0.1, 100, 128]
])

# 保存配置
np.savetxt(os.path.join(data_dir, 'config.txt'), config, fmt='%.3f', delimiter=',', 
           header='learning_rate,epochs,batch_size')
print("配置已保存")

# 加载配置
loaded_config = np.loadtxt(os.path.join(data_dir, 'config.txt'), delimiter=',', skiprows=1)
print("加载的配置:")
print(loaded_config)

# 10. 最佳实践
print("\n10. 最佳实践...")

print("""
文件I/O最佳实践：

1. 数据存储格式选择：
   - 对于NumPy数组：使用.npy格式（高效，保留数据类型）
   - 对于多个数组：使用.npz格式（压缩存储）
   - 对于文本数据：使用.txt或.csv格式（人类可读）
   - 对于大型数组：使用内存映射文件

2. 性能优化：
   - 对于频繁读写的小数据：使用.npy格式
   - 对于大型数据集：使用内存映射文件
   - 避免频繁的文件I/O操作
   - 批量读写数据

3. 数据安全：
   - 备份重要数据
   - 使用错误处理
   - 验证加载的数据

4. 代码组织：
   - 使用上下文管理器（with语句）处理文件
   - 封装I/O操作为函数
   - 统一文件路径管理
""")

# 11. 总结
print("\n" + "=" * 70)
print("文件I/O总结")
print("=" * 70)

print("""
NumPy文件I/O功能：

1. 主要文件格式：
   - .npy：NumPy专用二进制格式
   - .npz：压缩的NumPy数组集合
   - .txt：文本文件
   - .csv：逗号分隔值文件
   - 二进制文件：原始二进制格式

2. 核心函数：
   - np.save() / np.load()：保存/加载单个数组
   - np.savez() / np.savez_compressed()：保存多个数组
   - np.savetxt() / np.loadtxt()：保存/加载文本文件
   - np.fromfile() / np.ndarray.tofile()：二进制文件I/O
   - np.memmap()：内存映射文件

3. 应用场景：
   - 数据持久化
   - 模型参数保存
   - 配置文件读写
   - 大型数据集处理
   - 数据交换

4. 性能考虑：
   - 文件格式对读写速度的影响
   - 文件大小与存储效率
   - 内存使用
""")

print("=" * 70)
print("文件I/O演示完成！")
print("=" * 70)