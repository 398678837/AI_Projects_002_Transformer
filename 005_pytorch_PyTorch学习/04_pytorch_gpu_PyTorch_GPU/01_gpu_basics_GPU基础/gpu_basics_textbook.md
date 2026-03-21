# PyTorch GPU基础教材

## 第一章：GPU简介

### 1.1 什么是GPU

GPU（Graphics Processing Unit，图形处理单元）最初设计用于处理图形渲染任务，但由于其强大的并行计算能力，现在被广泛应用于深度学习等需要大规模并行计算的领域。

### 1.2 GPU的优势

- **并行计算能力**：GPU拥有大量的计算核心，可以同时处理多个任务
- **高内存带宽**：GPU内存带宽远高于CPU，适合处理大量数据
- **能效比**：GPU在相同功耗下可以提供更高的计算性能
- **深度学习优化**：现代GPU针对深度学习任务进行了专门优化

### 1.3 GPU vs CPU

| 特性 | CPU | GPU |
|------|-----|-----|
| 核心数量 | 少（通常4-32个） | 多（通常上千个） |
| 核心类型 | 复杂，适合串行任务 | 简单，适合并行任务 |
| 缓存大小 | 大 | 小 |
| 内存带宽 | 低 | 高 |
| 适合任务 | 复杂逻辑，串行计算 | 简单重复，并行计算 |

### 1.4 CUDA简介

CUDA（Compute Unified Device Architecture）是NVIDIA推出的并行计算平台和编程模型，允许开发者使用C/C++、Python等语言编写GPU加速的程序。

## 第二章：PyTorch GPU支持

### 2.1 检查GPU可用性

在使用GPU之前，我们需要先检查GPU是否可用：

```python
import torch

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
```

### 2.2 张量在CPU和GPU之间的移动

PyTorch中的张量可以在CPU和GPU之间自由移动：

```python
import torch

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
```

### 2.3 在GPU上执行张量操作

在GPU上执行张量操作可以显著提高计算速度：

```python
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # 创建两个GPU张量
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    
    print(f"张量a形状: {a.shape}, 设备: {a.device}")
    print(f"张量b形状: {b.shape}, 设备: {b.device}")
    
    # 执行矩阵乘法
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
```

### 2.4 模型在GPU上的使用

深度学习模型也可以在GPU上运行：

```python
import torch
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
```

## 第三章：多GPU使用

### 3.1 手动指定GPU

当有多个GPU可用时，我们可以手动指定使用哪个GPU：

```python
import torch

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
```

### 3.2 使用DataParallel

PyTorch提供了`DataParallel`类，使得模型可以在多个GPU上并行运行：

```python
import torch
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

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = SimpleModel()
    model = nn.DataParallel(model)
    model = model.to('cuda')
    
    print(f"DataParallel模型设备: {next(model.parameters()).device}")
    
    # 创建输入张量
    input_tensor = torch.randn(64, 100, device='cuda')
    
    # 前向传播
    output = model(input_tensor)
    print(f"输出张量形状: {output.shape}, 设备: {output.device}")
```

### 3.3 使用DistributedDataParallel

对于更复杂的分布式训练场景，PyTorch提供了`DistributedDataParallel`：

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

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

def train(rank, world_size):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # 创建模型
    model = SimpleModel().to(rank)
    # 包装模型
    ddp_model = DDP(model, device_ids=[rank])
    
    # 创建输入张量
    input_tensor = torch.randn(32, 100, device=rank)
    
    # 前向传播
    output = ddp_model(input_tensor)
    print(f"Rank {rank}, 输出张量形状: {output.shape}")
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

## 第四章：GPU内存管理

### 4.1 查看内存使用情况

在使用GPU时，我们需要关注内存使用情况，避免内存不足：

```python
import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated() / 1024**3
    reserved_memory = torch.cuda.memory_reserved() / 1024**3
    
    print(f"总内存: {total_memory:.2f} GB")
    print(f"已分配内存: {allocated_memory:.2f} GB")
    print(f"已保留内存: {reserved_memory:.2f} GB")
```

### 4.2 清空GPU缓存

当我们不再需要某些张量时，可以清空GPU缓存以释放内存：

```python
import torch

if torch.cuda.is_available():
    # 清空缓存
    torch.cuda.empty_cache()
    print("已清空GPU缓存")
```

### 4.3 内存管理最佳实践

1. **及时删除不需要的张量**：使用`del`关键字删除不再需要的张量
2. **使用上下文管理器**：在不需要GPU时释放内存
3. **批量处理数据**：使用适当的批量大小，避免一次性加载过多数据
4. **使用梯度累积**：在内存有限的情况下，可以使用梯度累积来模拟更大的批量大小
5. **监控内存使用**：定期检查内存使用情况，避免内存溢出

## 第五章：常见问题与解决方案

### 5.1 CUDA内存不足

**问题**：训练时出现`CUDA out of memory`错误

**解决方案**：
- 减小批量大小
- 使用梯度累积
- 清理不需要的张量
- 使用混合精度训练
- 考虑使用更大内存的GPU

### 5.2 设备不匹配

**问题**：运行时出现`Expected all tensors to be on the same device`错误

**解决方案**：
- 确保所有张量和模型都在同一设备上
- 使用`to()`方法将张量移动到正确的设备
- 检查数据加载器是否正确处理设备分配

### 5.3 多GPU训练速度慢

**问题**：多GPU训练速度没有预期的快

**解决方案**：
- 检查数据加载是否成为瓶颈
- 使用`DistributedDataParallel`代替`DataParallel`
- 确保批量大小足够大，充分利用GPU
- 检查GPU之间的通信是否高效

### 5.4 CUDA版本不匹配

**问题**：运行时出现`CUDA version mismatch`错误

**解决方案**：
- 确保PyTorch版本与CUDA版本匹配
- 重新安装与CUDA版本兼容的PyTorch
- 检查系统CUDA版本是否与PyTorch要求一致

## 第六章：代码优化技巧

### 6.1 使用device变量

```python
import torch

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 在代码中使用device
model = model.to(device)
input_tensor = input_tensor.to(device)
```

### 6.2 批量移动数据

```python
# 批量移动数据到GPU
batch = {k: v.to(device) for k, v in batch.items()}
```

### 6.3 使用inplace操作

```python
# 使用inplace操作减少内存使用
x = x.relu_()  # 注意末尾的下划线
```

### 6.4 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input_tensor)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.5 内存高效的数据加载

```python
from torch.utils.data import DataLoader

# 使用pin_memory和num_workers加速数据加载
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
```

## 第七章：性能评估

### 7.1 计算GPU利用率

```python
import torch
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # 创建大张量
    x = torch.randn(10000, 10000, device=device)
    
    # 执行操作并计时
    start = time.time()
    for i in range(100):
        x = torch.matmul(x, x)
        torch.cuda.synchronize()
    end = time.time()
    
    print(f"执行时间: {end - start:.4f} 秒")
    print(f"每秒操作数: {100 / (end - start):.2f}")
```

### 7.2 对比CPU和GPU性能

```python
import torch
import time

# 创建大张量
tensor_size = 5000

# CPU计算
x_cpu = torch.randn(tensor_size, tensor_size)
y_cpu = torch.randn(tensor_size, tensor_size)

start = time.time()
z_cpu = torch.matmul(x_cpu, y_cpu)
end = time.time()
print(f"CPU计算时间: {end - start:.4f} 秒")

# GPU计算
if torch.cuda.is_available():
    x_gpu = x_cpu.to('cuda')
    y_gpu = y_cpu.to('cuda')
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    end = time.time()
    print(f"GPU计算时间: {end - start:.4f} 秒")
    print(f"加速比: {(end - start) / (end - start):.2f}x")
```

## 第八章：习题

### 8.1 选择题

1. 以下哪种设备适合处理大规模并行计算任务？
   A. CPU
   B. GPU
   C. 内存
   D. 硬盘

2. 以下哪个方法可以将张量移动到GPU？
   A. `.cpu()`
   B. `.to('cuda')`
   C. `.numpy()`
   D. `.detach()`

3. 当有多个GPU可用时，以下哪种方法可以实现模型的并行训练？
   A. `nn.Sequential`
   B. `nn.Module`
   C. `nn.DataParallel`
   D. `nn.Linear`

### 8.2 填空题

1. CUDA是__________推出的并行计算平台和编程模型。
2. 在PyTorch中，可以使用__________方法检查CUDA是否可用。
3. 当训练时出现`CUDA out of memory`错误，可以通过__________来解决。

### 8.3 简答题

1. 解释GPU相比CPU的优势。
2. 如何在PyTorch中查看GPU内存使用情况？
3. 什么是混合精度训练？它有什么优势？

### 8.4 编程题

1. 编写代码检查系统中GPU的可用性和基本信息。

2. 编写代码比较CPU和GPU在执行矩阵乘法时的性能差异。

3. 编写代码实现一个简单的神经网络模型，并在GPU上训练它。

4. 编写代码使用多个GPU进行并行训练。

5. 编写代码监控GPU内存使用情况，并实现内存优化策略。

## 第九章：总结

### 9.1 知识回顾

1. **GPU基础**：了解GPU的基本概念和优势
2. **PyTorch GPU支持**：如何检查GPU可用性、在CPU和GPU之间移动张量
3. **在GPU上执行操作**：如何在GPU上执行张量操作和训练模型
4. **多GPU使用**：如何使用多个GPU进行并行计算
5. **GPU内存管理**：如何管理GPU内存，避免内存不足
6. **常见问题与解决方案**：如何解决GPU相关的常见问题
7. **代码优化技巧**：如何优化代码以充分利用GPU性能
8. **性能评估**：如何评估GPU性能和加速比

### 9.2 学习建议

1. **实践练习**：尝试在GPU上运行不同的模型和操作，观察性能差异
2. **理解原理**：理解GPU并行计算的原理和PyTorch的GPU支持机制
3. **内存管理**：学习如何有效管理GPU内存，避免内存不足
4. **多GPU训练**：尝试使用多个GPU进行并行训练，提高训练速度
5. **性能优化**：学习如何优化代码以充分利用GPU性能

### 9.3 进阶学习

1. **分布式训练**：学习使用`DistributedDataParallel`进行分布式训练
2. **混合精度训练**：学习使用混合精度训练提高训练速度和减少内存使用
3. **GPU编程**：学习CUDA编程，直接操作GPU
4. **模型量化**：学习模型量化技术，减少模型大小和内存使用
5. **推理优化**：学习模型推理优化技术，提高推理速度

通过本章的学习，您应该已经掌握了PyTorch中GPU的基本使用方法，可以开始在实际项目中应用GPU加速了。