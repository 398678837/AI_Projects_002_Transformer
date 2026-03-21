import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 版本:", torch.__version__)

# 1. 从Python列表创建张量
def create_from_list():
    print("\n1. 从Python列表创建张量:")
    # 一维张量
    data = [1, 2, 3, 4, 5]
    tensor = torch.tensor(data)
    print("一维张量:", tensor)
    print("张量形状:", tensor.shape)
    
    # 二维张量
    data_2d = [[1, 2, 3], [4, 5, 6]]
    tensor_2d = torch.tensor(data_2d)
    print("二维张量:", tensor_2d)
    print("张量形状:", tensor_2d.shape)
    
    # 三维张量
    data_3d = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    tensor_3d = torch.tensor(data_3d)
    print("三维张量:", tensor_3d)
    print("张量形状:", tensor_3d.shape)

# 2. 从NumPy数组创建张量
def create_from_numpy():
    print("\n2. 从NumPy数组创建张量:")
    # 一维NumPy数组
    np_array = np.array([1, 2, 3, 4, 5])
    tensor_from_np = torch.from_numpy(np_array)
    print("从NumPy创建的张量:", tensor_from_np)
    
    # 二维NumPy数组
    np_array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    tensor_from_np_2d = torch.from_numpy(np_array_2d)
    print("从NumPy创建的二维张量:", tensor_from_np_2d)
    
    # 注意：从NumPy创建的张量与原数组共享内存
    np_array[0] = 100
    print("修改NumPy数组后，张量也会变化:", tensor_from_np)

# 3. 创建特殊张量
def create_special_tensors():
    print("\n3. 创建特殊张量:")
    
    # 创建全零张量
    zeros = torch.zeros(2, 3)
    print("全零张量:", zeros)
    
    # 创建全一张量
    ones = torch.ones(3, 2)
    print("全一张量:", ones)
    
    # 创建单位矩阵
    eye = torch.eye(4)
    print("单位矩阵:", eye)
    
    # 创建指定值的张量
    full = torch.full((2, 2), 7)
    print("指定值的张量:", full)
    
    # 创建随机张量（均匀分布）
    rand = torch.rand(2, 3)
    print("随机张量（均匀分布）:", rand)
    
    # 创建随机张量（正态分布）
    randn = torch.randn(2, 3)
    print("随机张量（正态分布）:", randn)
    
    # 创建整数序列张量
    arange = torch.arange(0, 10, 2)
    print("整数序列张量:", arange)
    
    # 创建线性间隔张量
    linspace = torch.linspace(0, 1, 5)
    print("线性间隔张量:", linspace)

# 4. 张量的数据类型
def tensor_data_types():
    print("\n4. 张量的数据类型:")
    
    # 创建不同数据类型的张量
    tensor_float32 = torch.tensor([1, 2, 3], dtype=torch.float32)
    print("float32张量:", tensor_float32, "数据类型:", tensor_float32.dtype)
    
    tensor_int64 = torch.tensor([1, 2, 3], dtype=torch.int64)
    print("int64张量:", tensor_int64, "数据类型:", tensor_int64.dtype)
    
    tensor_bool = torch.tensor([True, False, True])
    print("bool张量:", tensor_bool, "数据类型:", tensor_bool.dtype)
    
    # 数据类型转换
    tensor_converted = tensor_int64.float()
    print("转换为float的张量:", tensor_converted, "数据类型:", tensor_converted.dtype)

# 5. 张量的设备
def tensor_devices():
    print("\n5. 张量的设备:")
    
    # 默认设备（CPU）
    tensor_cpu = torch.tensor([1, 2, 3])
    print("CPU张量设备:", tensor_cpu.device)
    
    # 检查是否有GPU
    if torch.cuda.is_available():
        # 移动到GPU
        tensor_gpu = tensor_cpu.to('cuda')
        print("GPU张量设备:", tensor_gpu.device)
    else:
        print("没有可用的GPU")

# 6. 张量的属性
def tensor_attributes():
    print("\n6. 张量的属性:")
    
    tensor = torch.rand(2, 3, 4)
    print("张量:", tensor)
    print("形状:", tensor.shape)
    print("维度:", tensor.ndim)
    print("元素数量:", tensor.numel())
    print("数据类型:", tensor.dtype)
    print("设备:", tensor.device)

# 7. 张量的可视化
def visualize_tensor():
    print("\n7. 张量的可视化:")
    
    # 创建一个二维张量
    tensor_2d = torch.rand(10, 10)
    
    # 可视化张量
    plt.figure(figsize=(8, 6))
    plt.imshow(tensor_2d.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('二维张量可视化')
    plt.savefig(os.path.join(images_dir, 'tensor_visualization.png'))
    plt.show()
    
    # 创建一个一维张量并可视化
    tensor_1d = torch.linspace(0, 1, 100)
    plt.figure(figsize=(8, 6))
    plt.plot(tensor_1d.numpy())
    plt.title('一维张量可视化')
    plt.xlabel('索引')
    plt.ylabel('值')
    plt.savefig(os.path.join(images_dir, 'tensor_1d_visualization.png'))
    plt.show()

# 8. 性能测试
def performance_test():
    print("\n8. 性能测试:")
    
    import time
    
    # 测试不同大小张量的创建时间
    sizes = [1000, 10000, 100000, 1000000]
    times = []
    
    for size in sizes:
        start = time.time()
        tensor = torch.rand(size)
        end = time.time()
        times.append(end - start)
        print(f"创建大小为 {size} 的张量耗时: {end - start:.6f} 秒")
    
    # 可视化性能
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('张量创建性能')
    plt.xlabel('张量大小')
    plt.ylabel('创建时间 (秒)')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'tensor_performance.png'))
    plt.show()

# 9. 实际应用示例
def practical_example():
    print("\n9. 实际应用示例:")
    
    # 示例1：线性回归中的权重和偏置
    print("示例1：线性回归中的权重和偏置")
    # 创建权重张量
    weights = torch.randn(3, 1, requires_grad=True)
    # 创建偏置张量
    bias = torch.randn(1, requires_grad=True)
    print("权重:", weights)
    print("偏置:", bias)
    
    # 示例2：图像数据
    print("\n示例2：图像数据")
    # 创建一个随机的RGB图像 (高度, 宽度, 通道)
    image = torch.rand(224, 224, 3)
    print("图像形状:", image.shape)
    print("图像数据类型:", image.dtype)
    
    # 示例3：批量数据
    print("\n示例3：批量数据")
    # 创建一个批量的图像数据 (批量大小, 通道, 高度, 宽度)
    batch = torch.rand(32, 3, 224, 224)
    print("批量数据形状:", batch.shape)

if __name__ == "__main__":
    print("PyTorch 张量创建演示")
    print("=" * 50)
    
    create_from_list()
    create_from_numpy()
    create_special_tensors()
    tensor_data_types()
    tensor_devices()
    tensor_attributes()
    visualize_tensor()
    performance_test()
    practical_example()
    
    print("\n" + "=" * 50)
    print("演示完成！")