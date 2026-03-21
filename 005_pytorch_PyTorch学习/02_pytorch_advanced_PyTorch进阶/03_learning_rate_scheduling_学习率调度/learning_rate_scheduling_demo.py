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

print("PyTorch 学习率调度演示")
print("=" * 50)

# 1. 基本学习率调度
def basic_learning_rate_scheduling():
    print("\n1. 基本学习率调度:")
    
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
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练模型，观察学习率变化
    learning_rates = []
    losses = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 记录学习率和损失
        learning_rates.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # 更新学习率
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{50}], Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss.item():.4f}')
    
    # 可视化学习率和损失
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(learning_rates)
    ax1.set_title('学习率变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('学习率')
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_title('损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'basic_lr_scheduling.png'))
    plt.show()

# 2. 不同类型的学习率调度器
def different_lr_schedulers():
    print("\n2. 不同类型的学习率调度器:")
    
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
    
    # 定义不同的学习率调度器
    schedulers = {
        'StepLR': optim.lr_scheduler.StepLR,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR
    }
    
    # 调度器参数
    scheduler_params = {
        'StepLR': {'step_size': 10, 'gamma': 0.1},
        'MultiStepLR': {'milestones': [10, 20, 30], 'gamma': 0.1},
        'ExponentialLR': {'gamma': 0.95},
        'CosineAnnealingLR': {'T_max': 50}
    }
    
    all_learning_rates = {}
    all_losses = {}
    
    for name, scheduler_class in schedulers.items():
        # 创建模型、损失函数和优化器
        model = SimpleNet()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # 创建调度器
        if name == 'MultiStepLR':
            scheduler = scheduler_class(optimizer, **scheduler_params[name])
        else:
            scheduler = scheduler_class(optimizer, **scheduler_params[name])
        
        learning_rates = []
        losses = []
        
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # 记录学习率和损失
            learning_rates.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())
            
            # 更新学习率
            scheduler.step()
        
        all_learning_rates[name] = learning_rates
        all_losses[name] = losses
        print(f"{name}, 最终学习率: {learning_rates[-1]:.6f}, 最终损失: {losses[-1]:.4f}")
    
    # 可视化不同调度器的学习率变化
    plt.figure(figsize=(10, 6))
    for name, learning_rates in all_learning_rates.items():
        plt.plot(learning_rates, label=name)
    plt.title('不同学习率调度器的学习率变化')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'different_schedulers.png'))
    plt.show()

# 3. 学习率预热
def learning_rate_warmup():
    print("\n3. 学习率预热:")
    
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
    
    # 学习率预热
    warmup_epochs = 5
    initial_lr = 0.01
    target_lr = 0.1
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练模型，使用学习率预热
    learning_rates = []
    losses = []
    
    for epoch in range(50):
        # 学习率预热
        if epoch < warmup_epochs:
            # 线性预热
            current_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif epoch == warmup_epochs:
            # 预热结束，设置目标学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 记录学习率和损失
        learning_rates.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{50}], Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss.item():.4f}')
    
    # 可视化学习率和损失
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(learning_rates)
    ax1.set_title('学习率变化（带预热）')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('学习率')
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_title('损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'lr_warmup.png'))
    plt.show()

# 4. 自定义学习率调度器
def custom_lr_scheduler():
    print("\n4. 自定义学习率调度器:")
    
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
    
    # 自定义学习率调度器
    class CustomLRScheduler:
        def __init__(self, optimizer, initial_lr, decay_rate):
            self.optimizer = optimizer
            self.initial_lr = initial_lr
            self.decay_rate = decay_rate
            self.step_count = 0
        
        def step(self):
            self.step_count += 1
            # 自定义学习率衰减策略
            lr = self.initial_lr * (1 / (1 + self.decay_rate * self.step_count))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 创建自定义调度器
    scheduler = CustomLRScheduler(optimizer, initial_lr=0.1, decay_rate=0.01)
    
    # 生成数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 训练模型
    learning_rates = []
    losses = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 记录学习率和损失
        learning_rates.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        # 更新学习率
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{50}], Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss.item():.4f}')
    
    # 可视化学习率和损失
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(learning_rates)
    ax1.set_title('学习率变化（自定义调度器）')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('学习率')
    ax1.grid(True)
    
    ax2.plot(losses)
    ax2.set_title('损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'custom_scheduler.png'))
    plt.show()

# 5. 学习率调度与模型训练
def lr_scheduling_in_training():
    print("\n5. 学习率调度与模型训练:")
    
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
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # 划分训练集和测试集
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    # 创建模型、损失函数和优化器
    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # 训练模型
    num_epochs = 100
    train_losses = []
    test_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        train_loss = criterion(outputs, train_y)
        train_loss.backward()
        optimizer.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_X)
            test_loss = criterion(test_outputs, test_y)
        
        # 更新学习率
        scheduler.step(test_loss)
        
        # 记录数据
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 可视化结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
    ax1.plot(learning_rates)
    ax1.set_title('学习率变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('学习率')
    ax1.grid(True)
    
    ax2.plot(train_losses, label='Train Loss')
    ax2.plot(test_losses, label='Test Loss')
    ax2.set_title('损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('损失')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(range(len(learning_rates)), learning_rates, 'r-', label='Learning Rate')
    ax3.set_title('学习率与损失关系')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('学习率')
    ax3.twinx().plot(range(len(test_losses)), test_losses, 'b-', label='Test Loss')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'lr_scheduling_training.png'))
    plt.show()

# 6. 学习率调度与不同优化器
def lr_scheduling_with_different_optimizers():
    print("\n6. 学习率调度与不同优化器:")
    
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
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # 划分训练集和测试集
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    # 测试不同的优化器
    optimizers = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop
    }
    
    all_train_losses = {}
    all_test_losses = {}
    all_learning_rates = {}
    
    for name, opt_class in optimizers.items():
        # 创建模型、损失函数和优化器
        model = SimpleNet()
        criterion = nn.MSELoss()
        
        if name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=0.1)
        else:
            optimizer = opt_class(model.parameters(), lr=0.001)
        
        # 创建学习率调度器
        if name == 'SGD':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        train_losses = []
        test_losses = []
        learning_rates = []
        
        for epoch in range(100):
            # 训练
            model.train()
            optimizer.zero_grad()
            outputs = model(train_X)
            train_loss = criterion(outputs, train_y)
            train_loss.backward()
            optimizer.step()
            
            # 测试
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_X)
                test_loss = criterion(test_outputs, test_y)
            
            # 更新学习率
            scheduler.step()
            
            # 记录数据
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            learning_rates.append(optimizer.param_groups[0]['lr'])
        
        all_train_losses[name] = train_losses
        all_test_losses[name] = test_losses
        all_learning_rates[name] = learning_rates
        print(f"{name}, 最终学习率: {learning_rates[-1]:.6f}, 最终测试损失: {test_losses[-1]:.4f}")
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    for name, learning_rates in all_learning_rates.items():
        ax1.plot(learning_rates, label=name)
    ax1.set_title('不同优化器的学习率变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('学习率')
    ax1.legend()
    ax1.grid(True)
    
    for name, test_losses in all_test_losses.items():
        ax2.plot(test_losses, label=name)
    ax2.set_title('不同优化器的测试损失变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('测试损失')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'optimizers_lr_scheduling.png'))
    plt.show()

if __name__ == "__main__":
    basic_learning_rate_scheduling()
    different_lr_schedulers()
    learning_rate_warmup()
    custom_lr_scheduler()
    lr_scheduling_in_training()
    lr_scheduling_with_different_optimizers()
    
    print("\n" + "=" * 50)
    print("演示完成！")