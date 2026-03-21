import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 神经网络演示")
print("=" * 50)

# 1. 基本神经网络
def basic_neural_network():
    print("\n1. 基本神经网络:")
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=1, n_redundant=10, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = NeuralNetwork(X.shape[1], 50, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 计算训练准确率
        with torch.no_grad():
            train_pred = (outputs > 0.5).float()
            train_accuracy = (train_pred == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            test_pred = (test_outputs > 0.5).float()
            test_accuracy = (test_pred == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'neural_network_basic.png'))
    plt.show()

# 2. 多分类神经网络
def multi_class_neural_network():
    print("\n2. 多分类神经网络:")
    
    # 生成多分类数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=1, n_redundant=10, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class MultiClassNeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MultiClassNeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = MultiClassNeuralNetwork(X.shape[1], 50, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 计算训练准确率
        with torch.no_grad():
            _, train_pred = torch.max(outputs, 1)
            train_accuracy = (train_pred == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            _, test_pred = torch.max(test_outputs, 1)
            test_accuracy = (test_pred == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'neural_network_multiclass.png'))
    plt.show()

# 3. 深层神经网络
def deep_neural_network():
    print("\n3. 深层神经网络:")
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_clusters_per_class=1, n_redundant=10, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class DeepNeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super(DeepNeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            
            # 添加输入层到第一个隐藏层
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(nn.ReLU())
            
            # 添加中间隐藏层
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                self.layers.append(nn.ReLU())
            
            # 添加输出层
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.layers.append(nn.Sigmoid())
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = DeepNeuralNetwork(X.shape[1], [100, 50, 25], 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 计算训练准确率
        with torch.no_grad():
            train_pred = (outputs > 0.5).float()
            train_accuracy = (train_pred == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            test_pred = (test_outputs > 0.5).float()
            test_accuracy = (test_pred == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'neural_network_deep.png'))
    plt.show()

# 4. 神经网络与真实数据集
def neural_network_real_data():
    print("\n4. 神经网络与真实数据集:")
    
    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 转换为张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = NeuralNetwork(X.shape[1], 50, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 计算训练准确率
        with torch.no_grad():
            train_pred = (outputs > 0.5).float()
            train_accuracy = (train_pred == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            test_pred = (test_outputs > 0.5).float()
            test_accuracy = (test_pred == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title('乳腺癌数据集神经网络')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'neural_network_breast_cancer.png'))
    plt.show()

# 5. 正则化神经网络
def regularized_neural_network():
    print("\n5. 正则化神经网络:")
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=50, n_classes=2, n_clusters_per_class=1, n_redundant=30, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class RegularizedNeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegularizedNeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    # 创建模型、损失函数和优化器
    model = RegularizedNeuralNetwork(X.shape[1], 100, 1)
    criterion = nn.BCELoss()
    # 添加L2正则化
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 计算训练准确率
        with torch.no_grad():
            train_pred = (outputs > 0.5).float()
            train_accuracy = (train_pred == y_train).sum().item() / y_train.size(0)
            train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            test_pred = (test_outputs > 0.5).float()
            test_accuracy = (test_pred == y_test).sum().item() / y_test.size(0)
            test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(test_losses, label='测试损失')
    ax1.set_title('正则化神经网络')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(test_accuracies, label='测试准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'neural_network_regularized.png'))
    plt.show()

if __name__ == "__main__":
    basic_neural_network()
    multi_class_neural_network()
    deep_neural_network()
    neural_network_real_data()
    regularized_neural_network()
    
    print("\n" + "=" * 50)
    print("演示完成！")