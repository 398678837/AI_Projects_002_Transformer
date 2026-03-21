import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 数据加载演示")
print("=" * 50)

# 1. 基本数据加载
def basic_data_loader():
    print("\n1. 基本数据加载:")
    
    # 创建示例数据
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # 创建数据集
    dataset = data.TensorDataset(X, y)
    
    # 创建数据加载器
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 遍历数据加载器
    print("数据加载器批次:")
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        print(f"批次 {batch_idx+1}: X形状={batch_X.shape}, y形状={batch_y.shape}")
        if batch_idx == 2:  # 只显示前3个批次
            break

# 2. 自定义数据集
def custom_dataset():
    print("\n2. 自定义数据集:")
    
    class CustomDataset(data.Dataset):
        def __init__(self, size, transform=None):
            self.size = size
            self.transform = transform
            # 生成随机数据
            self.data = torch.randn(size, 10)
            self.targets = torch.randn(size, 1)
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            target = self.targets[idx]
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample, target
    
    # 创建自定义数据集
    dataset = CustomDataset(1000)
    print(f"数据集大小: {len(dataset)}")
    
    # 获取单个样本
    sample, target = dataset[0]
    print(f"单个样本形状: {sample.shape}, 目标形状: {target.shape}")
    
    # 创建数据加载器
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 遍历数据加载器
    print("数据加载器批次:")
    for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
        print(f"批次 {batch_idx+1}: X形状={batch_X.shape}, y形状={batch_y.shape}")
        if batch_idx == 2:  # 只显示前3个批次
            break

# 3. 数据变换
def data_transforms():
    print("\n3. 数据变换:")
    
    # 定义变换
    class NormalizeTransform:
        def __call__(self, sample):
            return (sample - sample.mean()) / sample.std()
    
    class AddNoiseTransform:
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level
        
        def __call__(self, sample):
            return sample + torch.randn_like(sample) * self.noise_level
    
    # 创建组合变换
    from torchvision import transforms
    transform = transforms.Compose([
        NormalizeTransform(),
        AddNoiseTransform()
    ])
    
    # 创建数据集
    class CustomDataset(data.Dataset):
        def __init__(self, size, transform=None):
            self.size = size
            self.transform = transform
            self.data = torch.randn(size, 10)
            self.targets = torch.randn(size, 1)
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            target = self.targets[idx]
            
            if self.transform:
                sample = self.transform(sample)
            
            return sample, target
    
    # 创建数据集和数据加载器
    dataset = CustomDataset(1000, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 测试变换效果
    sample, target = dataset[0]
    print(f"变换后样本: {sample}")
    print(f"变换后样本均值: {sample.mean():.4f}, 标准差: {sample.std():.4f}")

# 4. 数据加载器参数
def dataloader_parameters():
    print("\n4. 数据加载器参数:")
    
    # 创建示例数据
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = data.TensorDataset(X, y)
    
    # 测试不同的batch_size
    print("不同batch_size的效果:")
    for batch_size in [16, 32, 64]:
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"batch_size={batch_size}, 批次数={len(dataloader)}")
    
    # 测试shuffle参数
    print("\nshuffle参数的效果:")
    dataloader_no_shuffle = data.DataLoader(dataset, batch_size=32, shuffle=False)
    dataloader_shuffle = data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 获取第一个批次
    for batch_X, batch_y in dataloader_no_shuffle:
        print(f"shuffle=False 第一个批次前5个样本:")
        print(batch_X[:5, 0])
        break
    
    for batch_X, batch_y in dataloader_shuffle:
        print(f"shuffle=True 第一个批次前5个样本:")
        print(batch_X[:5, 0])
        break

# 5. 多进程数据加载
def multiprocessing_data_loading():
    print("\n5. 多进程数据加载:")
    
    # 创建大型数据集
    X = torch.randn(10000, 100)
    y = torch.randn(10000, 1)
    dataset = data.TensorDataset(X, y)
    
    # 测试不同的num_workers
    import time
    
    print("不同num_workers的加载时间:")
    for num_workers in [0, 2, 4, 8]:
        dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers)
        
        start = time.time()
        for batch_X, batch_y in dataloader:
            pass
        end = time.time()
        
        print(f"num_workers={num_workers}, 加载时间: {end - start:.4f}秒")

# 6. 数据加载器与模型训练
def dataloader_training():
    print("\n6. 数据加载器与模型训练:")
    
    # 创建数据集
    class CustomDataset(data.Dataset):
        def __init__(self, size):
            self.size = size
            # 生成线性回归数据
            self.X = torch.randn(size, 1)
            self.y = 2 * self.X + 3 + torch.randn(size, 1) * 0.1
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # 创建训练集和测试集
    train_dataset = CustomDataset(800)
    test_dataset = CustomDataset(200)
    
    # 创建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 定义模型
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 创建模型、损失函数和优化器
    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 50
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        
        # 测试
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'dataloader_training_loss.png'))
    plt.show()

# 7. 真实数据加载
def real_data_loading():
    print("\n7. 真实数据加载:")
    
    # 尝试加载MNIST数据集（如果可用）
    try:
        from torchvision import datasets, transforms
        
        # 定义变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载MNIST数据集
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 创建数据加载器
        train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 查看数据
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"批次 {batch_idx+1}: 图像形状={images.shape}, 标签形状={labels.shape}")
            print(f"标签示例: {labels[:10]}")
            break
            
    except ImportError:
        print("torchvision 未安装，跳过真实数据加载演示")

# 8. 数据加载性能优化
def data_loading_optimization():
    print("\n8. 数据加载性能优化:")
    
    # 创建大型数据集
    X = torch.randn(50000, 100)
    y = torch.randn(50000, 1)
    dataset = data.TensorDataset(X, y)
    
    # 测试不同的pin_memory设置
    import time
    
    print("不同pin_memory设置的加载时间:")
    for pin_memory in [False, True]:
        dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory)
        
        start = time.time()
        for batch_X, batch_y in dataloader:
            pass
        end = time.time()
        
        print(f"pin_memory={pin_memory}, 加载时间: {end - start:.4f}秒")

if __name__ == "__main__":
    basic_data_loader()
    custom_dataset()
    data_transforms()
    dataloader_parameters()
    multiprocessing_data_loading()
    dataloader_training()
    real_data_loading()
    data_loading_optimization()
    
    print("\n" + "=" * 50)
    print("演示完成！")