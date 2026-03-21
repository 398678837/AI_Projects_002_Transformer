import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 创建images目录（如果不存在）
images_dir = os.path.join(script_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

print("PyTorch 自动微分演示")
print("=" * 50)

# 1. 基本自动微分
def basic_autograd():
    print("\n1. 基本自动微分:")
    
    # 创建一个需要梯度的张量
    x = torch.tensor(2.0, requires_grad=True)
    print("x:", x)
    print("x.requires_grad:", x.requires_grad)
    
    # 定义一个函数 y = x^2
    y = x ** 2
    print("y:", y)
    
    # 计算梯度
    y.backward()
    print("x.grad:", x.grad)  # 应该是 4.0
    
    # 清除梯度
    x.grad.zero_()
    print("清除梯度后 x.grad:", x.grad)

# 2. 多变量自动微分
def multivariable_autograd():
    print("\n2. 多变量自动微分:")
    
    # 创建两个需要梯度的张量
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # 定义一个函数 z = x^2 + y^2
    z = x ** 2 + y ** 2
    print("z:", z)
    
    # 计算梯度
    z.backward()
    print("x.grad:", x.grad)  # 应该是 4.0
    print("y.grad:", y.grad)  # 应该是 6.0

# 3. 复杂计算图
def complex_computation_graph():
    print("\n3. 复杂计算图:")
    
    # 创建输入张量
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    
    # 定义复杂计算
    z = x ** 2 * y + y
    print("z:", z)
    
    # 计算梯度
    z.backward()
    print("x.grad:", x.grad)  # 应该是 2 * x * y = 2 * 2 * 3 = 12
    print("y.grad:", y.grad)  # 应该是 x^2 + 1 = 4 + 1 = 5

# 4. 向量和矩阵的自动微分
def vector_matrix_autograd():
    print("\n4. 向量和矩阵的自动微分:")
    
    # 创建向量
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print("x:", x)
    
    # 计算向量的范数
    y = torch.norm(x)
    print("y:", y)
    
    # 计算梯度
    y.backward()
    print("x.grad:", x.grad)  # 应该是 x / ||x||
    
    # 矩阵的自动微分
    print("\n矩阵的自动微分:")
    A = torch.randn(2, 3, requires_grad=True)
    B = torch.randn(3, 2, requires_grad=True)
    C = A @ B
    print("C:", C)
    
    # 计算梯度
    C.sum().backward()
    print("A.grad:", A.grad)
    print("B.grad:", B.grad)

# 5. 梯度累积
def gradient_accumulation():
    print("\n5. 梯度累积:")
    
    x = torch.tensor(1.0, requires_grad=True)
    
    # 第一次前向传播
    y = x ** 2
    y.backward()
    print("第一次 backward 后 x.grad:", x.grad)  # 2
    
    # 第二次前向传播（不清除梯度）
    y = x ** 2
    y.backward()
    print("第二次 backward 后 x.grad:", x.grad)  # 4（梯度累积）
    
    # 清除梯度后再计算
    x.grad.zero_()
    y = x ** 2
    y.backward()
    print("清除梯度后 x.grad:", x.grad)  # 2

# 6. 禁用梯度计算
def disable_grad():
    print("\n6. 禁用梯度计算:")
    
    x = torch.tensor(1.0, requires_grad=True)
    
    # 正常计算梯度
    y = x ** 2
    y.backward()
    print("正常计算后 x.grad:", x.grad)
    
    # 禁用梯度计算
    x.grad.zero_()
    with torch.no_grad():
        y = x ** 2
        print("禁用梯度计算时 y.requires_grad:", y.requires_grad)
    
    # 尝试在禁用梯度计算后计算梯度
    try:
        y.backward()
        print("x.grad:", x.grad)
    except RuntimeError as e:
        print("错误:", e)

# 7. 自定义函数的自动微分
def custom_function():
    print("\n7. 自定义函数的自动微分:")
    
    # 定义一个自定义函数
    def f(x):
        return x ** 3 + 2 * x
    
    x = torch.tensor(2.0, requires_grad=True)
    y = f(x)
    print("f(2):", y)
    
    # 计算梯度
    y.backward()
    print("f'(2):", x.grad)  # 应该是 3x^2 + 2 = 3*4 + 2 = 14

# 8. 线性回归中的自动微分
def linear_regression_autograd():
    print("\n8. 线性回归中的自动微分:")
    
    # 生成数据
    X = torch.randn(100, 1)
    y = 2 * X + 3 + torch.randn(100, 1) * 0.1
    
    # 初始化参数
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    
    # 训练参数
    learning_rate = 0.01
    num_epochs = 100
    
    losses = []
    
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = X @ w + b
        
        # 计算损失
        loss = torch.mean((y_pred - y) ** 2)
        losses.append(loss.item())
        
        # 计算梯度
        loss.backward()
        
        # 更新参数
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
        
        # 清除梯度
        w.grad.zero_()
        b.grad.zero_()
    
    print("训练完成")
    print("w:", w.item())
    print("b:", b.item())
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('线性回归损失')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'linear_regression_loss.png'))
    plt.show()

# 9. 神经网络中的自动微分
def neural_network_autograd():
    print("\n9. 神经网络中的自动微分:")
    
    # 生成数据
    X = torch.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
    
    # 定义神经网络
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = torch.nn.Linear(2, 10)
            self.fc2 = torch.nn.Linear(10, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            return x
    
    # 初始化模型
    model = NeuralNetwork()
    
    # 定义损失函数和优化器
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # 训练模型
    num_epochs = 100
    losses = []
    
    for epoch in range(num_epochs):
        # 前向传播
        y_pred = model(X)
        
        # 计算损失
        loss = criterion(y_pred, y)
        losses.append(loss.item())
        
        # 计算梯度
        optimizer.zero_grad()
        loss.backward()
        
        # 更新参数
        optimizer.step()
    
    print("训练完成")
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('神经网络损失')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'neural_network_loss.png'))
    plt.show()

# 10. 性能测试
def performance_test():
    print("\n10. 性能测试:")
    
    import time
    
    # 测试不同大小张量的自动微分性能
    sizes = [100, 1000, 5000]
    times = []
    
    for size in sizes:
        # 创建张量
        x = torch.randn(size, size, requires_grad=True)
        
        # 前向传播
        start = time.time()
        y = x.sum()
        
        # 反向传播
        y.backward()
        end = time.time()
        
        times.append(end - start)
        print(f"大小为 {size}x{size} 的张量自动微分耗时: {end - start:.6f} 秒")
    
    # 可视化性能
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('自动微分性能')
    plt.xlabel('张量大小')
    plt.ylabel('耗时 (秒)')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'autograd_performance.png'))
    plt.show()

if __name__ == "__main__":
    basic_autograd()
    multivariable_autograd()
    complex_computation_graph()
    vector_matrix_autograd()
    gradient_accumulation()
    disable_grad()
    custom_function()
    linear_regression_autograd()
    neural_network_autograd()
    performance_test()
    
    print("\n" + "=" * 50)
    print("演示完成！")