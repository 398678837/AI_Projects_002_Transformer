import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 梯度裁剪演示")
print("=" * 50)

# 1. 梯度爆炸问题演示
def gradient_explosion_demo():
    print("\n1. 梯度爆炸问题演示:")
    
    # 创建一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练模型，观察梯度变化
    gradients = []
    
    for i in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        gradients.append(grad_norm)
        
        print(f"迭代 {i+1}, 损失: {loss.item():.4f}, 梯度范数: {grad_norm:.4f}")
        
        # 更新参数
        optimizer.step()
    
    # 可视化梯度范数
    plt.figure(figsize=(8, 6))
    plt.plot(gradients)
    plt.title('梯度范数变化（无梯度裁剪）')
    plt.xlabel('迭代次数')
    plt.ylabel('梯度范数')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'gradient_explosion.png'))
    plt.show()

# 2. 基本梯度裁剪
def basic_gradient_clipping():
    print("\n2. 基本梯度裁剪:")
    
    # 创建一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练模型，使用梯度裁剪
    gradients = []
    
    for i in range(10):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        gradients.append(grad_norm)
        
        print(f"迭代 {i+1}, 损失: {loss.item():.4f}, 梯度范数: {grad_norm:.4f}")
        
        # 更新参数
        optimizer.step()
    
    # 可视化梯度范数
    plt.figure(figsize=(8, 6))
    plt.plot(gradients)
    plt.title('梯度范数变化（有梯度裁剪）')
    plt.xlabel('迭代次数')
    plt.ylabel('梯度范数')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'gradient_clipping.png'))
    plt.show()

# 3. 不同裁剪阈值的效果
def different_clipping_thresholds():
    print("\n3. 不同裁剪阈值的效果:")
    
    # 创建一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 测试不同的裁剪阈值
    thresholds = [0.1, 0.5, 1.0, 5.0]
    all_gradients = []
    
    for threshold in thresholds:
        # 创建模型、损失函数和优化器
        model = SimpleNet()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        gradients = []
        
        for i in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=threshold)
            
            # 计算梯度范数
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            gradients.append(grad_norm)
            
            # 更新参数
            optimizer.step()
        
        all_gradients.append(gradients)
        print(f"裁剪阈值 {threshold}, 最终梯度范数: {gradients[-1]:.4f}")
    
    # 可视化不同裁剪阈值的效果
    plt.figure(figsize=(8, 6))
    for i, threshold in enumerate(thresholds):
        plt.plot(all_gradients[i], label=f'阈值={threshold}')
    plt.title('不同裁剪阈值的梯度范数变化')
    plt.xlabel('迭代次数')
    plt.ylabel('梯度范数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'different_thresholds.png'))
    plt.show()

# 4. 梯度裁剪与学习率的关系
def gradient_clipping_and_learning_rate():
    print("\n4. 梯度裁剪与学习率的关系:")
    
    # 创建一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 测试不同的学习率
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    all_gradients = []
    
    for lr in learning_rates:
        # 创建模型、损失函数和优化器
        model = SimpleNet()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        gradients = []
        
        for i in range(10):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 计算梯度范数
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            gradients.append(grad_norm)
            
            # 更新参数
            optimizer.step()
        
        all_gradients.append(gradients)
        print(f"学习率 {lr}, 最终梯度范数: {gradients[-1]:.4f}")
    
    # 可视化不同学习率的效果
    plt.figure(figsize=(8, 6))
    for i, lr in enumerate(learning_rates):
        plt.plot(all_gradients[i], label=f'学习率={lr}')
    plt.title('不同学习率的梯度范数变化（有梯度裁剪）')
    plt.xlabel('迭代次数')
    plt.ylabel('梯度范数')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'learning_rates.png'))
    plt.show()

# 5. 梯度裁剪在RNN中的应用
def gradient_clipping_in_rnn():
    print("\n5. 梯度裁剪在RNN中的应用:")
    
    # 创建一个简单的RNN
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_size)
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型、损失函数和优化器
    model = SimpleRNN(input_size=10, hidden_size=100, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 生成序列数据
    seq_length = 50
    X = torch.randn(32, seq_length, 10)
    y = torch.randn(32, 1)
    
    # 训练模型，使用梯度裁剪
    gradients = []
    losses = []
    
    for i in range(20):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        gradients.append(grad_norm)
        losses.append(loss.item())
        
        print(f"迭代 {i+1}, 损失: {loss.item():.4f}, 梯度范数: {grad_norm:.4f}")
        
        # 更新参数
        optimizer.step()
    
    # 可视化梯度范数和损失
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(gradients)
    ax1.set_title('梯度范数变化')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('梯度范数')
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_title('损失变化')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'rnn_gradient_clipping.png'))
    plt.show()

# 6. 梯度裁剪与不同优化器
def gradient_clipping_with_different_optimizers():
    print("\n6. 梯度裁剪与不同优化器:")
    
    # 创建一个简单的网络
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 测试不同的优化器
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }
    
    all_gradients = {}
    all_losses = {}
    
    for name, opt_class in optimizers.items():
        # 创建模型、损失函数和优化器
        model = SimpleNet()
        criterion = nn.MSELoss()
        optimizer = opt_class(model.parameters(), lr=0.01)
        
        gradients = []
        losses = []
        
        for i in range(20):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 计算梯度范数
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            gradients.append(grad_norm)
            losses.append(loss.item())
            
            # 更新参数
            optimizer.step()
        
        all_gradients[name] = gradients
        all_losses[name] = losses
        print(f"优化器 {name}, 最终损失: {losses[-1]:.4f}, 最终梯度范数: {gradients[-1]:.4f}")
    
    # 可视化不同优化器的效果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    for name, gradients in all_gradients.items():
        ax1.plot(gradients, label=name)
    ax1.set_title('不同优化器的梯度范数变化')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('梯度范数')
    ax1.legend()
    ax1.grid(True)
    
    for name, losses in all_losses.items():
        ax2.plot(losses, label=name)
    ax2.set_title('不同优化器的损失变化')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'optimizers_comparison.png'))
    plt.show()

# 7. 梯度裁剪的性能影响
def gradient_clipping_performance():
    print("\n7. 梯度裁剪的性能影响:")
    
    # 创建一个大型网络
    class LargeNet(nn.Module):
        def __init__(self):
            super(LargeNet, self).__init__()
            self.fc1 = nn.Linear(100, 1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc3 = nn.Linear(1000, 1000)
            self.fc4 = nn.Linear(1000, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    # 生成数据
    X = torch.randn(32, 100)
    y = torch.randn(32, 1)
    
    # 测试有无梯度裁剪的性能
    import time
    
    # 无梯度裁剪
    model1 = LargeNet()
    criterion = nn.MSELoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
    
    start = time.time()
    for i in range(100):
        optimizer1.zero_grad()
        outputs = model1(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer1.step()
    end = time.time()
    print(f"无梯度裁剪: {end - start:.4f}秒")
    
    # 有梯度裁剪
    model2 = LargeNet()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    
    start = time.time()
    for i in range(100):
        optimizer2.zero_grad()
        outputs = model2(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer2.step()
    end = time.time()
    print(f"有梯度裁剪: {end - start:.4f}秒")

if __name__ == "__main__":
    gradient_explosion_demo()
    basic_gradient_clipping()
    different_clipping_thresholds()
    gradient_clipping_and_learning_rate()
    gradient_clipping_in_rnn()
    gradient_clipping_with_different_optimizers()
    gradient_clipping_performance()
    
    print("\n" + "=" * 50)
    print("演示完成！")