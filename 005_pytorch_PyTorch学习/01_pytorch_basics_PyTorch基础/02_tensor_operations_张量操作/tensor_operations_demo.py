import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 张量操作演示")
print("=" * 50)

# 1. 张量索引和切片
def tensor_indexing():
    print("\n1. 张量索引和切片:")
    
    # 创建一个3x4的张量
    tensor = torch.tensor([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12]])
    print("原始张量:")
    print(tensor)
    
    # 索引单个元素
    print("\n索引单个元素:")
    print("tensor[0, 0]:", tensor[0, 0])  # 第一行第一列
    print("tensor[1, 2]:", tensor[1, 2])  # 第二行第三列
    
    # 切片操作
    print("\n切片操作:")
    print("tensor[:, 0]:", tensor[:, 0])  # 第一列
    print("tensor[0, :]:", tensor[0, :])  # 第一行
    print("tensor[1:3, 1:3]:", tensor[1:3, 1:3])  # 第二、三行，第二、三列
    
    # 布尔索引
    print("\n布尔索引:")
    mask = tensor > 5
    print("mask:")
    print(mask)
    print("tensor[mask]:", tensor[mask])

# 2. 张量形状操作
def tensor_shape_operations():
    print("\n2. 张量形状操作:")
    
    # 创建一个张量
    tensor = torch.randn(2, 3, 4)
    print("原始张量形状:", tensor.shape)
    
    # reshape
    print("\nreshape:")
    reshaped = tensor.reshape(2, 12)
    print("reshape(2, 12):", reshaped.shape)
    
    # view
    print("\nview:")
    viewed = tensor.view(2, 12)
    print("view(2, 12):", viewed.shape)
    
    # squeeze 和 unsqueeze
    print("\nsqueeze 和 unsqueeze:")
    tensor_2d = torch.randn(1, 4)
    print("原始形状:", tensor_2d.shape)
    squeezed = tensor_2d.squeeze()
    print("squeeze:", squeezed.shape)
    unsqueezed = squeezed.unsqueeze(0)
    print("unsqueeze(0):", unsqueezed.shape)
    
    # transpose
    print("\ntranspose:")
    tensor_2d = torch.randn(2, 3)
    print("原始形状:", tensor_2d.shape)
    transposed = tensor_2d.transpose(0, 1)
    print("transpose(0, 1):", transposed.shape)
    
    # permute
    print("\npermute:")
    tensor_3d = torch.randn(2, 3, 4)
    print("原始形状:", tensor_3d.shape)
    permuted = tensor_3d.permute(2, 0, 1)
    print("permute(2, 0, 1):", permuted.shape)

# 3. 张量数学运算
def tensor_math_operations():
    print("\n3. 张量数学运算:")
    
    # 创建两个张量
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    
    # 基本运算
    print("\n基本运算:")
    print("a + b:", a + b)
    print("a - b:", a - b)
    print("a * b:", a * b)
    print("a / b:", a / b)
    print("a ** 2:", a ** 2)
    
    # 矩阵乘法
    print("\n矩阵乘法:")
    c = torch.tensor([[1, 2], [3, 4]])
    d = torch.tensor([[5, 6], [7, 8]])
    print("c.matmul(d):")
    print(c.matmul(d))
    print("c @ d:")
    print(c @ d)
    
    # 广播机制
    print("\n广播机制:")
    e = torch.tensor([[1, 2, 3], [4, 5, 6]])
    f = torch.tensor([10, 20, 30])
    print("e + f:")
    print(e + f)

# 4. 张量统计操作
def tensor_statistics():
    print("\n4. 张量统计操作:")
    
    # 创建一个张量
    tensor = torch.randn(3, 4)
    print("原始张量:")
    print(tensor)
    
    # 基本统计
    print("\n基本统计:")
    print("均值:", tensor.mean())
    print("总和:", tensor.sum())
    print("最大值:", tensor.max())
    print("最小值:", tensor.min())
    print("标准差:", tensor.std())
    
    # 沿特定维度统计
    print("\n沿特定维度统计:")
    print("沿维度0的均值:", tensor.mean(dim=0))
    print("沿维度1的总和:", tensor.sum(dim=1))
    print("沿维度0的最大值:", tensor.max(dim=0))
    print("沿维度1的最小值:", tensor.min(dim=1))

# 5. 张量比较操作
def tensor_comparison():
    print("\n5. 张量比较操作:")
    
    # 创建两个张量
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([3, 3, 3, 3, 3])
    
    print("a:", a)
    print("b:", b)
    
    # 比较操作
    print("\n比较操作:")
    print("a == b:", a == b)
    print("a != b:", a != b)
    print("a > b:", a > b)
    print("a < b:", a < b)
    print("a >= b:", a >= b)
    print("a <= b:", a <= b)
    
    # 其他比较函数
    print("\n其他比较函数:")
    print("torch.all(a == b):", torch.all(a == b))
    print("torch.any(a > b):", torch.any(a > b))

# 6. 张量类型转换
def tensor_type_conversion():
    print("\n6. 张量类型转换:")
    
    # 创建一个张量
    tensor = torch.tensor([1, 2, 3, 4, 5])
    print("原始张量:", tensor)
    print("原始类型:", tensor.dtype)
    
    # 类型转换
    print("\n类型转换:")
    print("转换为float:", tensor.float())
    print("转换为double:", tensor.double())
    print("转换为int:", tensor.int())
    print("转换为bool:", tensor.bool())
    
    # 转换为NumPy数组
    print("\n转换为NumPy数组:")
    numpy_array = tensor.numpy()
    print("NumPy数组:", numpy_array)
    print("NumPy数组类型:", numpy_array.dtype)

# 7. 张量的设备移动
def tensor_device_movement():
    print("\n7. 张量的设备移动:")
    
    # 创建一个CPU张量
    tensor = torch.tensor([1, 2, 3, 4, 5])
    print("原始设备:", tensor.device)
    
    # 检查是否有GPU
    if torch.cuda.is_available():
        # 移动到GPU
        tensor_gpu = tensor.to('cuda')
        print("移动到GPU后:", tensor_gpu.device)
        
        # 移动回CPU
        tensor_cpu = tensor_gpu.to('cpu')
        print("移动回CPU后:", tensor_cpu.device)
    else:
        print("没有可用的GPU")

# 8. 张量的内存管理
def tensor_memory_management():
    print("\n8. 张量的内存管理:")
    
    # 创建一个张量
    tensor = torch.randn(1000, 1000)
    print("张量大小:", tensor.size())
    print("元素数量:", tensor.numel())
    print("内存占用 (MB):", tensor.element_size() * tensor.numel() / 1024 / 1024)
    
    # 释放内存
    del tensor
    print("张量已删除")

# 9. 张量的高级操作
def tensor_advanced_operations():
    print("\n9. 张量的高级操作:")
    
    # 创建一个张量
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("原始张量:")
    print(tensor)
    
    # 求迹
    print("\n求迹:", tensor.trace())
    
    # 求范数
    print("\n求范数:")
    print("L1范数:", tensor.norm(1))
    print("L2范数:", tensor.norm(2))
    
    # 排序
    print("\n排序:")
    sorted_tensor, indices = torch.sort(tensor, dim=1)
    print("排序后的张量:")
    print(sorted_tensor)
    print("索引:")
    print(indices)
    
    # 唯一值
    print("\n唯一值:")
    unique_values = torch.unique(tensor)
    print("唯一值:", unique_values)

# 10. 性能测试
def performance_test():
    print("\n10. 性能测试:")
    
    import time
    
    # 测试不同大小张量的运算性能
    sizes = [100, 1000, 5000]
    times = []
    
    for size in sizes:
        # 创建两个随机张量
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # 测试矩阵乘法性能
        start = time.time()
        c = a @ b
        end = time.time()
        
        times.append(end - start)
        print(f"大小为 {size}x{size} 的矩阵乘法耗时: {end - start:.6f} 秒")
    
    # 可视化性能
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('矩阵乘法性能')
    plt.xlabel('矩阵大小')
    plt.ylabel('耗时 (秒)')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'matrix_multiplication_performance.png'))
    plt.show()

# 11. 实际应用示例
def practical_example():
    print("\n11. 实际应用示例:")
    
    # 示例1：线性回归
    print("示例1：线性回归")
    # 创建输入和目标
    X = torch.randn(100, 3)
    y = torch.randn(100, 1)
    # 创建权重和偏置
    w = torch.randn(3, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    # 前向传播
    y_pred = X @ w + b
    print("预测形状:", y_pred.shape)
    
    # 示例2：图像处理
    print("\n示例2：图像处理")
    # 创建一个批量的图像数据 (批量大小, 通道, 高度, 宽度)
    batch = torch.rand(32, 3, 224, 224)
    print("批量形状:", batch.shape)
    # 计算每个通道的均值
    channel_means = batch.mean(dim=(0, 2, 3))
    print("通道均值:", channel_means)

if __name__ == "__main__":
    tensor_indexing()
    tensor_shape_operations()
    tensor_math_operations()
    tensor_statistics()
    tensor_comparison()
    tensor_type_conversion()
    tensor_device_movement()
    tensor_memory_management()
    tensor_advanced_operations()
    performance_test()
    practical_example()
    
    print("\n" + "=" * 50)
    print("演示完成！")