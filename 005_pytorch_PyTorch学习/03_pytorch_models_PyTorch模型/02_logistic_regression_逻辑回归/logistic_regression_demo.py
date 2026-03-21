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

print("PyTorch 逻辑回归演示")
print("=" * 50)

# 1. 基本逻辑回归
def basic_logistic_regression():
    print("\n1. 基本逻辑回归:")
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    # 创建模型、损失函数和优化器
    model = LogisticRegression(X.shape[1])
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
        print(f'测试准确率: {accuracy:.4f}')
    
    # 可视化决策边界
    def plot_decision_boundary(model, X, y):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        with torch.no_grad():
            Z = model(grid).numpy()
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y.numpy().ravel(), edgecolors='k')
        plt.title('逻辑回归决策边界')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.savefig(os.path.join(images_dir, 'logistic_regression_decision_boundary.png'))
        plt.show()
    
    plot_decision_boundary(model, X.numpy(), y.numpy())

# 2. 多分类逻辑回归
def multi_class_logistic_regression():
    print("\n2. 多分类逻辑回归:")
    
    # 生成多分类数据
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, n_clusters_per_class=1, n_redundant=0, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # 多分类任务中，标签不需要one-hot编码
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class MultiClassLogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MultiClassLogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.linear(x)  # 多分类任务中，不需要sigmoid，使用CrossEntropyLoss会自动应用softmax
    
    # 创建模型、损失函数和优化器
    model = MultiClassLogisticRegression(X.shape[1], 3)
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        _, y_pred_class = torch.max(y_pred, 1)
        accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
        print(f'测试准确率: {accuracy:.4f}')

# 3. 逻辑回归与真实数据集
def logistic_regression_real_data():
    print("\n3. 逻辑回归与真实数据集:")
    
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
    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    # 创建模型、损失函数和优化器
    model = LogisticRegression(X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
        print(f'测试准确率: {accuracy:.4f}')
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('乳腺癌数据集逻辑回归')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'logistic_regression_breast_cancer.png'))
    plt.show()

# 4. 正则化逻辑回归
def regularized_logistic_regression():
    print("\n4. 正则化逻辑回归:")
    
    # 生成数据
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2, n_clusters_per_class=1, n_redundant=10, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义模型
    class LogisticRegression(nn.Module):
        def __init__(self, input_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.linear(x))
    
    # 创建模型、损失函数和优化器
    model = LogisticRegression(X.shape[1])
    criterion = nn.BCELoss()
    # 添加L2正则化
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    
    # 训练模型
    num_epochs = 1000
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_class = (y_pred > 0.5).float()
        accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
        print(f'测试准确率: {accuracy:.4f}')

if __name__ == "__main__":
    basic_logistic_regression()
    multi_class_logistic_regression()
    logistic_regression_real_data()
    regularized_logistic_regression()
    
    print("\n" + "=" * 50)
    print("演示完成！")