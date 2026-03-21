import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch GPU优化演示")
print("=" * 50)

# 1. 准备数据
def prepare_data():
    print("\n1. 准备数据:")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    
    return trainset, testset

# 2. 定义模型
def define_model():
    print("\n2. 定义模型:")
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = Net()
    print("模型结构:")
    print(model)
    
    return model

# 3. 数据加载器优化
def data_loader_optimization(trainset, testset):
    print("\n3. 数据加载器优化:")
    
    # 基本数据加载器
    basic_trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    basic_testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    
    # 优化的数据加载器
    optimized_trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=128, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2
    )
    
    optimized_testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2
    )
    
    print("基本数据加载器:")
    print(f"  批量大小: {basic_trainloader.batch_size}")
    print(f"  工作进程数: {basic_trainloader.num_workers}")
    print(f"  Pin memory: {basic_trainloader.pin_memory}")
    
    print("优化的数据加载器:")
    print(f"  批量大小: {optimized_trainloader.batch_size}")
    print(f"  工作进程数: {optimized_trainloader.num_workers}")
    print(f"  Pin memory: {optimized_trainloader.pin_memory}")
    print(f"  Prefetch factor: {optimized_trainloader.prefetch_factor}")
    
    return basic_trainloader, basic_testloader, optimized_trainloader, optimized_testloader

# 4. 模型优化
def model_optimization():
    print("\n4. 模型优化:")
    
    # 原始模型
    class OriginalNet(nn.Module):
        def __init__(self):
            super(OriginalNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 优化的模型
    class OptimizedNet(nn.Module):
        def __init__(self):
            super(OptimizedNet, self).__init__()
            # 使用批归一化
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.bn1 = nn.BatchNorm2d(6)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.bn3 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.bn4 = nn.BatchNorm1d(84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
            x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 16 * 5 * 5)
            x = nn.functional.relu(self.bn3(self.fc1(x)))
            x = nn.functional.relu(self.bn4(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    original_model = OriginalNet()
    optimized_model = OptimizedNet()
    
    print("原始模型:")
    print(original_model)
    
    print("\n优化的模型:")
    print(optimized_model)
    
    return original_model, optimized_model

# 5. 训练函数
def train_model(model, trainloader, testloader, device, num_epochs=5):
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 训练参数
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练
        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss = test_loss / len(testloader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    end_time = time.time()
    print(f"训练完成！总时间: {end_time - start_time:.2f} 秒")
    
    return train_losses, test_losses, train_accuracies, test_accuracies, end_time - start_time

# 6. 内存优化
def memory_optimization():
    print("\n6. 内存优化:")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试不同批量大小的内存使用
    batch_sizes = [32, 64, 128, 256]
    memory_usage = []
    
    for batch_size in batch_sizes:
        # 创建随机张量
        x = torch.randn(batch_size, 3, 32, 32, device=device)
        
        # 计算内存使用
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        memory_usage.append(allocated_memory)
        
        print(f"批量大小: {batch_size}, 内存使用: {allocated_memory:.2f} MB")
        
        # 清理
        del x
        torch.cuda.empty_cache()
    
    # 可视化内存使用
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, memory_usage, marker='o')
    plt.title('批量大小与内存使用关系')
    plt.xlabel('批量大小')
    plt.ylabel('内存使用 (MB)')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'memory_usage.png'))
    plt.show()
    
    return batch_sizes, memory_usage

# 7. 推理优化
def inference_optimization():
    print("\n7. 推理优化:")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = define_model().to(device)
    
    # 创建输入张量
    input_tensor = torch.randn(1, 3, 32, 32, device=device)
    
    # 常规推理
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            output = model(input_tensor)
    torch.cuda.synchronize()
    normal_time = time.time() - start_time
    print(f"常规推理时间: {normal_time:.4f} 秒")
    
    # 使用torch.jit.trace优化
    traced_model = torch.jit.trace(model, input_tensor)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            output = traced_model(input_tensor)
    torch.cuda.synchronize()
    jit_time = time.time() - start_time
    print(f"JIT优化推理时间: {jit_time:.4f} 秒")
    print(f"加速比: {normal_time / jit_time:.2f}x")
    
    # 使用torch.jit.script优化
    scripted_model = torch.jit.script(model)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(1000):
            output = scripted_model(input_tensor)
    torch.cuda.synchronize()
    script_time = time.time() - start_time
    print(f"Script优化推理时间: {script_time:.4f} 秒")
    print(f"加速比: {normal_time / script_time:.2f}x")
    
    return normal_time, jit_time, script_time

# 8. 可视化结果
def visualize_results(results, title):
    print(f"\n8. 可视化{title}结果:")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 损失曲线
    ax1.plot(results['train_losses'], label='训练损失')
    ax1.plot(results['test_losses'], label='测试损失')
    ax1.set_title(f'{title}损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(results['train_accuracies'], label='训练准确率')
    ax2.plot(results['test_accuracies'], label='测试准确率')
    ax2.set_title(f'{title}准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, f'{title.replace(" ", "_")}_results.png'))
    plt.show()

if __name__ == "__main__":
    # 准备数据
    trainset, testset = prepare_data()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        # 数据加载器优化
        basic_trainloader, basic_testloader, optimized_trainloader, optimized_testloader = data_loader_optimization(trainset, testset)
        
        # 测试基本数据加载器
        print("\n测试基本数据加载器:")
        model = define_model()
        basic_results = {}
        basic_results['train_losses'], basic_results['test_losses'], basic_results['train_accuracies'], basic_results['test_accuracies'], basic_time = train_model(model, basic_trainloader, basic_testloader, device)
        visualize_results(basic_results, "基本数据加载器")
        
        # 测试优化的数据加载器
        print("\n测试优化的数据加载器:")
        model = define_model()
        optimized_results = {}
        optimized_results['train_losses'], optimized_results['test_losses'], optimized_results['train_accuracies'], optimized_results['test_accuracies'], optimized_time = train_model(model, optimized_trainloader, optimized_testloader, device)
        visualize_results(optimized_results, "优化数据加载器")
        
        print(f"\n数据加载器优化效果:")
        print(f"基本数据加载器训练时间: {basic_time:.2f} 秒")
        print(f"优化数据加载器训练时间: {optimized_time:.2f} 秒")
        print(f"加速比: {basic_time / optimized_time:.2f}x")
        
        # 模型优化
        original_model, optimized_model = model_optimization()
        
        # 测试原始模型
        print("\n测试原始模型:")
        original_results = {}
        original_results['train_losses'], original_results['test_losses'], original_results['train_accuracies'], original_results['test_accuracies'], original_time = train_model(original_model, optimized_trainloader, optimized_testloader, device)
        visualize_results(original_results, "原始模型")
        
        # 测试优化的模型
        print("\n测试优化的模型:")
        optimized_model_results = {}
        optimized_model_results['train_losses'], optimized_model_results['test_losses'], optimized_model_results['train_accuracies'], optimized_model_results['test_accuracies'], optimized_model_time = train_model(optimized_model, optimized_trainloader, optimized_testloader, device)
        visualize_results(optimized_model_results, "优化模型")
        
        print(f"\n模型优化效果:")
        print(f"原始模型训练时间: {original_time:.2f} 秒")
        print(f"优化模型训练时间: {optimized_model_time:.2f} 秒")
        print(f"加速比: {original_time / optimized_model_time:.2f}x")
        
        # 内存优化
        memory_optimization()
        
        # 推理优化
        inference_optimization()
    else:
        print("CUDA不可用，无法进行GPU优化演示")
    
    print("\n" + "=" * 50)
    print("演示完成！")