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

print("PyTorch 线性回归演示")
print("=" * 50)

# 1. 基本线性回归
def basic_linear_regression():
    print("\n1. 基本线性回归:")
    
    # 生成数据
    x = torch.linspace(0, 10, 100).unsqueeze(1)
    y = 2 * x + 1 + torch.randn(100, 1) * 0.5
    
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
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 测试模型
    with torch.no_grad():
        predicted = model(x)
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.scatter(x.numpy(), y.numpy(), label='实际数据')
    plt.plot(x.numpy(), predicted.numpy(), 'r-', label='预测结果')
    plt.title('线性回归')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'linear_regression_basic.png'))
    plt.show()

# 2. 多元线性回归
def multiple_linear_regression():
    print("\n2. 多元线性回归:")
    
    # 生成数据
    n_samples = 100
    n_features = 3
    x = torch.randn(n_samples, n_features)
    # 真实参数
    weights = torch.tensor([2.5, -1.5, 0.8])
    bias = 1.2
    y = x @ weights + bias + torch.randn(n_samples) * 0.5
    y = y.unsqueeze(1)
    
    # 定义模型
    class MultipleLinearRegression(nn.Module):
        def __init__(self, input_size):
            super(MultipleLinearRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 创建模型、损失函数和优化器
    model = MultipleLinearRegression(n_features)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 打印真实参数和预测参数
    print("\n真实参数:")
    print(f'权重: {weights}')
    print(f'偏置: {bias}')
    print("\n预测参数:")
    print(f'权重: {model.linear.weight.data.numpy()[0]}')
    print(f'偏置: {model.linear.bias.data.numpy()[0]}')

# 3. 多项式回归
def polynomial_regression():
    print("\n3. 多项式回归:")
    
    # 生成数据
    x = torch.linspace(-3, 3, 100).unsqueeze(1)
    y = x**3 + 2*x**2 - 3*x + 1 + torch.randn(100, 1) * 2
    
    # 特征转换（添加多项式特征）
    def polynomial_features(x, degree):
        features = []
        for i in range(1, degree+1):
            features.append(x**i)
        return torch.cat(features, dim=1)
    
    degree = 3
    x_poly = polynomial_features(x, degree)
    
    # 定义模型
    class PolynomialRegression(nn.Module):
        def __init__(self, input_size):
            super(PolynomialRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 创建模型、损失函数和优化器
    model = PolynomialRegression(degree)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 10000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x_poly)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 测试模型
    with torch.no_grad():
        predicted = model(x_poly)
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.scatter(x.numpy(), y.numpy(), label='实际数据')
    plt.plot(x.numpy(), predicted.numpy(), 'r-', label='预测结果')
    plt.title(f'{degree}阶多项式回归')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, f'polynomial_regression_degree_{degree}.png'))
    plt.show()

# 4. 正则化线性回归
def regularized_linear_regression():
    print("\n4. 正则化线性回归:")
    
    # 生成数据
    n_samples = 50
    n_features = 10
    x = torch.randn(n_samples, n_features)
    # 真实参数（大部分为0）
    weights = torch.zeros(n_features)
    weights[0] = 2.0
    weights[1] = -1.5
    bias = 1.0
    y = x @ weights + bias + torch.randn(n_samples) * 0.5
    y = y.unsqueeze(1)
    
    # 定义模型
    class RegularizedLinearRegression(nn.Module):
        def __init__(self, input_size):
            super(RegularizedLinearRegression, self).__init__()
            self.linear = nn.Linear(input_size, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # 创建模型、损失函数和优化器
    model = RegularizedLinearRegression(n_features)
    criterion = nn.MSELoss()
    # 添加L2正则化
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    
    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 打印真实参数和预测参数
    print("\n真实参数:")
    print(f'权重: {weights}')
    print(f'偏置: {bias}')
    print("\n预测参数:")
    print(f'权重: {model.linear.weight.data.numpy()[0]}')
    print(f'偏置: {model.linear.bias.data.numpy()[0]}')

# 5. 线性回归与真实数据集
def linear_regression_real_data():
    print("\n5. 线性回归与真实数据集:")
    
    # 使用波士顿房价数据集（注意：在较新版本的scikit-learn中已移除）
    try:
        from sklearn.datasets import load_boston
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        # 加载数据
        boston = load_boston()
        X, y = boston.data, boston.target
        
        # 数据预处理
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 转换为张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 定义模型
        class LinearRegression(nn.Module):
            def __init__(self, input_size):
                super(LinearRegression, self).__init__()
                self.linear = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # 创建模型、损失函数和优化器
        model = LinearRegression(X.shape[1])
        criterion = nn.MSELoss()
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
        
        # 可视化损失
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.title('波士顿房价数据集线性回归')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(images_dir, 'linear_regression_boston.png'))
        plt.show()
        
    except ImportError:
        print("波士顿房价数据集在当前scikit-learn版本中不可用，使用合成数据代替")
        
        # 生成合成数据
        n_samples = 506
        n_features = 13
        X = torch.randn(n_samples, n_features)
        weights = torch.randn(n_features)
        bias = torch.randn(1)
        y = X @ weights + bias + torch.randn(n_samples) * 5
        y = y.unsqueeze(1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 定义模型
        class LinearRegression(nn.Module):
            def __init__(self, input_size):
                super(LinearRegression, self).__init__()
                self.linear = nn.Linear(input_size, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        # 创建模型、损失函数和优化器
        model = LinearRegression(n_features)
        criterion = nn.MSELoss()
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
        
        # 可视化损失
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.title('合成数据集线性回归')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(images_dir, 'linear_regression_synthetic.png'))
        plt.show()

if __name__ == "__main__":
    basic_linear_regression()
    multiple_linear_regression()
    polynomial_regression()
    regularized_linear_regression()
    linear_regression_real_data()
    
    print("\n" + "=" * 50)
    print("演示完成！")