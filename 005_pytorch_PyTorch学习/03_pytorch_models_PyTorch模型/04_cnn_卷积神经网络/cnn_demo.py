import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 卷积神经网络演示")
print("=" * 50)

# 1. 基本CNN模型
def basic_cnn():
    print("\n1. 基本CNN模型:")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = SimpleCNN()
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 3, 32, 32)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 2. LeNet-5模型
def lenet5():
    print("\n2. LeNet-5模型:")
    
    class LeNet5(nn.Module):
        def __init__(self):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 16 * 4 * 4)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # 创建模型
    model = LeNet5()
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 1, 32, 32)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 3. ResNet模型
def resnet():
    print("\n3. ResNet模型:")
    
    # 残差块
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            #  shortcut连接
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(residual)
            out = self.relu(out)
            return out
    
    # ResNet模型
    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        
        def _make_layer(self, block, out_channels, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_channels, out_channels, stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    
    # 创建ResNet18模型
    def resnet18():
        return ResNet(ResidualBlock, [2, 2, 2, 2])
    
    # 创建模型
    model = resnet18()
    print("ResNet18模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 3, 32, 32)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 4. CNN训练
def cnn_training():
    print("\n4. CNN训练:")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载CIFAR-10数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 定义CNN模型
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
    
    # 创建模型、损失函数和优化器
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # 训练模型
    num_epochs = 10
    train_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                train_losses.append(running_loss / 2000)
                running_loss = 0.0
    
    print('训练完成！')
    
    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'测试准确率: {100 * correct / total}%')
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses)
    plt.title('CNN训练损失')
    plt.xlabel('批次')
    plt.ylabel('损失')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'cnn_training_loss.png'))
    plt.show()

# 5. 预训练模型
def pretrained_model():
    print("\n5. 预训练模型:")
    
    # 加载预训练的ResNet18模型
    from torchvision import models
    
    resnet18 = models.resnet18(pretrained=True)
    print("ResNet18预训练模型加载成功")
    
    # 测试模型
    input_tensor = torch.randn(1, 3, 224, 224)
    output = resnet18(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 微调模型
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 10)  # 10个类别
    print("模型微调完成")

# 6. CNN可视化
def cnn_visualization():
    print("\n6. CNN可视化:")
    
    # 定义一个简单的CNN模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 8 * 8)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = SimpleCNN()
    
    # 可视化卷积核
    def visualize_kernels(model):
        conv1_weights = model.conv1.weight.data
        print(f"conv1权重形状: {conv1_weights.shape}")
        
        # 可视化前6个卷积核
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for i in range(6):
            ax = axes[i//3, i%3]
            kernel = conv1_weights[i].permute(1, 2, 0).numpy()
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())  # 归一化
            ax.imshow(kernel)
            ax.set_title(f'卷积核 {i+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'cnn_kernels.png'))
        plt.show()
    
    # 可视化特征图
    def visualize_feature_maps(model, input_tensor):
        # 创建特征提取器
        class FeatureExtractor(nn.Module):
            def __init__(self, model):
                super(FeatureExtractor, self).__init__()
                self.features = nn.Sequential(*list(model.children())[:-3])
            
            def forward(self, x):
                return self.features(x)
        
        extractor = FeatureExtractor(model)
        features = extractor(input_tensor)
        print(f"特征图形状: {features.shape}")
        
        # 可视化前16个特征图
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            ax = axes[i//4, i%4]
            feature = features[0, i].detach().numpy()
            feature = (feature - feature.min()) / (feature.max() - feature.min())  # 归一化
            ax.imshow(feature, cmap='gray')
            ax.set_title(f'特征图 {i+1}')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'cnn_feature_maps.png'))
        plt.show()
    
    # 可视化卷积核
    visualize_kernels(model)
    
    # 可视化特征图
    input_tensor = torch.randn(1, 3, 32, 32)
    visualize_feature_maps(model, input_tensor)

if __name__ == "__main__":
    basic_cnn()
    lenet5()
    resnet()
    cnn_training()
    pretrained_model()
    cnn_visualization()
    
    print("\n" + "=" * 50)
    print("演示完成！")