import torch
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch GPU基础演示")
print("=" * 50)

# 1. 检查GPU可用性
def check_gpu_availability():
    print("\n1. 检查GPU可用性:")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # 获取GPU数量
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        # 获取GPU名称
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 获取当前GPU索引
        current_device = torch.cuda.current_device()
        print(f"当前GPU索引: {current_device}")
        print(f"当前GPU名称: {torch.cuda.get_device_name(current_device)}")
        
        # 获取GPU内存信息
        total_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        print(f"GPU总内存: {total_memory:.2f} GB")
        
        # 计算剩余内存
        torch.cuda.empty_cache()  # 清空缓存
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"已分配内存: {allocated_memory:.2f} GB")
        print(f"已保留内存: {reserved_memory:.2f} GB")
        print(f"剩余内存: {total_memory - reserved_memory:.2f} GB")

# 2. 张量在CPU和GPU之间的移动
def tensor_movement():
    print("\n2. 张量在CPU和GPU之间的移动:")
    
    # 创建CPU张量
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f"CPU张量: {cpu_tensor}")
    print(f"CPU张量设备: {cpu_tensor.device}")
    
    if torch.cuda.is_available():
        # 将张量移动到GPU
        gpu_tensor = cpu_tensor.to('cuda')
        print(f"GPU张量: {gpu_tensor}")
        print(f"GPU张量设备: {gpu_tensor.device}")
        
        # 将张量从GPU移回CPU
        cpu_tensor_again = gpu_tensor.to('cpu')
        print(f"移回CPU的张量: {cpu_tensor_again}")
        print(f"移回CPU的张量设备: {cpu_tensor_again.device}")
        
        # 使用.cuda()方法
        gpu_tensor2 = cpu_tensor.cuda()
        print(f"使用.cuda()方法创建的GPU张量: {gpu_tensor2}")
        print(f"使用.cuda()方法创建的GPU张量设备: {gpu_tensor2.device}")
        
        # 使用.to(device)方法
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_tensor3 = cpu_tensor.to(device)
        print(f"使用.to(device)方法创建的GPU张量: {gpu_tensor3}")
        print(f"使用.to(device)方法创建的GPU张量设备: {gpu_tensor3.device}")

# 3. 在GPU上执行张量操作
def gpu_tensor_operations():
    print("\n3. 在GPU上执行张量操作:")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        # 创建两个GPU张量
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        print(f"张量a形状: {a.shape}, 设备: {a.device}")
        print(f"张量b形状: {b.shape}, 设备: {b.device}")
        
        # 执行矩阵乘法
        import time
        
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 等待GPU操作完成
        end = time.time()
        
        print(f"GPU矩阵乘法时间: {end - start:.4f} 秒")
        print(f"结果张量形状: {c.shape}, 设备: {c.device}")
        
        # 对比CPU执行时间
        a_cpu = a.to('cpu')
        b_cpu = b.to('cpu')
        
        start = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        end = time.time()
        
        print(f"CPU矩阵乘法时间: {end - start:.4f} 秒")
        print(f"CPU结果张量形状: {c_cpu.shape}, 设备: {c_cpu.device}")

# 4. 模型在GPU上的使用
def model_on_gpu():
    print("\n4. 模型在GPU上的使用:")
    
    import torch.nn as nn
    
    # 定义一个简单的模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = SimpleModel()
    print(f"模型初始设备: {next(model.parameters()).device}")
    
    if torch.cuda.is_available():
        # 将模型移动到GPU
        model = model.to('cuda')
        print(f"模型移动后设备: {next(model.parameters()).device}")
        
        # 创建输入张量
        input_tensor = torch.randn(32, 100, device='cuda')
        
        # 前向传播
        output = model(input_tensor)
        print(f"输出张量形状: {output.shape}, 设备: {output.device}")

# 5. 多GPU使用
def multi_gpu_usage():
    print("\n5. 多GPU使用:")
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"检测到多个GPU: {torch.cuda.device_count()}个")
        
        # 方法1: 手动指定GPU
        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')
        
        # 在不同GPU上创建张量
        tensor0 = torch.randn(100, 100, device=device0)
        tensor1 = torch.randn(100, 100, device=device1)
        
        print(f"张量0设备: {tensor0.device}")
        print(f"张量1设备: {tensor1.device}")
        
        # 注意：不同GPU上的张量不能直接运算，需要先移动到同一设备
        # tensor2 = tensor0 + tensor1  # 这会报错
        # 正确做法
        tensor1_on_0 = tensor1.to(device0)
        tensor2 = tensor0 + tensor1_on_0
        print(f"结果张量设备: {tensor2.device}")
        
        # 方法2: 使用DataParallel
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(100, 50)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(50, 10)
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = SimpleModel()
        model = nn.DataParallel(model)
        model = model.to('cuda')
        
        print(f"DataParallel模型设备: {next(model.parameters()).device}")
        
        # 创建输入张量
        input_tensor = torch.randn(64, 100, device='cuda')
        
        # 前向传播
        output = model(input_tensor)
        print(f"输出张量形状: {output.shape}, 设备: {output.device}")
    else:
        print("没有检测到多个GPU，或CUDA不可用")

# 6. GPU内存管理
def gpu_memory_management():
    print("\n6. GPU内存管理:")
    
    if torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        print("已清空GPU缓存")
        
        # 查看内存使用情况
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        
        print(f"总内存: {total_memory:.2f} GB")
        print(f"已分配内存: {allocated_memory:.2f} GB")
        print(f"已保留内存: {reserved_memory:.2f} GB")
        
        # 创建大张量
        large_tensor = torch.randn(10000, 10000, device='cuda')
        print("创建了大张量")
        
        # 再次查看内存使用情况
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"创建大张量后:")
        print(f"已分配内存: {allocated_memory:.2f} GB")
        print(f"已保留内存: {reserved_memory:.2f} GB")
        
        # 删除张量
        del large_tensor
        print("已删除大张量")
        
        # 清空缓存
        torch.cuda.empty_cache()
        print("已清空GPU缓存")
        
        # 再次查看内存使用情况
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"删除大张量并清空缓存后:")
        print(f"已分配内存: {allocated_memory:.2f} GB")
        print(f"已保留内存: {reserved_memory:.2f} GB")

if __name__ == "__main__":
    check_gpu_availability()
    tensor_movement()
    gpu_tensor_operations()
    model_on_gpu()
    multi_gpu_usage()
    gpu_memory_management()
    
    print("\n" + "=" * 50)
    print("演示完成！")