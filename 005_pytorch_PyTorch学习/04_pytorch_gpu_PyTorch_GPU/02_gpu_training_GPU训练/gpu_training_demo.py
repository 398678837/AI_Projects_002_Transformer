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

print("PyTorch GPU训练演示")
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"类别: {classes}")
    
    return trainloader, testloader, classes

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

# 3. 训练模型（CPU vs GPU）
def train_model(model, trainloader, testloader, device):
    print(f"\n3. 训练模型（{device}）:")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 训练参数
    num_epochs = 10
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
    
    return train_losses, test_losses, train_accuracies, test_accuracies

# 4. 混合精度训练
def mixed_precision_training(model, trainloader, testloader, device):
    print("\n4. 混合精度训练:")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 使用混合精度
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # 训练参数
    num_epochs = 10
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
            
            # 使用autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 使用scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
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
            with autocast():
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
    print(f"混合精度训练完成！总时间: {end_time - start_time:.2f} 秒")
    
    return train_losses, test_losses, train_accuracies, test_accuracies

# 5. 梯度累积
def gradient_accumulation(model, trainloader, testloader, device):
    print("\n5. 梯度累积:")
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # 训练参数
    num_epochs = 10
    accumulation_steps = 4  # 梯度累积步数
    batch_size = trainloader.batch_size
    effective_batch_size = batch_size * accumulation_steps
    
    print(f"原始批量大小: {batch_size}")
    print(f"梯度累积步数: {accumulation_steps}")
    print(f"有效批量大小: {effective_batch_size}")
    
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
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 缩放损失以保持与批量大小的一致性
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 累积梯度
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 确保最后一个批次的梯度也被更新
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
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
    print(f"梯度累积训练完成！总时间: {end_time - start_time:.2f} 秒")
    
    return train_losses, test_losses, train_accuracies, test_accuracies

# 6. 可视化结果
def visualize_results(train_losses, test_losses, train_accuracies, test_accuracies, title):
    print(f"\n6. 可视化{title}结果:")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 损失曲线
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title(f'{title}损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
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
    trainloader, testloader, classes = prepare_data()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    if torch.cuda.is_available():
        # GPU训练
        model = define_model()
        gpu_train_losses, gpu_test_losses, gpu_train_accuracies, gpu_test_accuracies = train_model(model, trainloader, testloader, device)
        visualize_results(gpu_train_losses, gpu_test_losses, gpu_train_accuracies, gpu_test_accuracies, "GPU训练")
        
        # 混合精度训练
        model = define_model()
        mixed_train_losses, mixed_test_losses, mixed_train_accuracies, mixed_test_accuracies = mixed_precision_training(model, trainloader, testloader, device)
        visualize_results(mixed_train_losses, mixed_test_losses, mixed_train_accuracies, mixed_test_accuracies, "混合精度训练")
        
        # 梯度累积训练
        model = define_model()
        grad_acc_train_losses, grad_acc_test_losses, grad_acc_train_accuracies, grad_acc_test_accuracies = gradient_accumulation(model, trainloader, testloader, device)
        visualize_results(grad_acc_train_losses, grad_acc_test_losses, grad_acc_train_accuracies, grad_acc_test_accuracies, "梯度累积训练")
    else:
        # CPU训练
        model = define_model()
        cpu_train_losses, cpu_test_losses, cpu_train_accuracies, cpu_test_accuracies = train_model(model, trainloader, testloader, device)
        visualize_results(cpu_train_losses, cpu_test_losses, cpu_train_accuracies, cpu_test_accuracies, "CPU训练")
    
    print("\n" + "=" * 50)
    print("演示完成！")