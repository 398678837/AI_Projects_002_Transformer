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

print("PyTorch 神经网络模块演示")
print("=" * 50)

# 1. 基本神经网络模块
def basic_nn_modules():
    print("\n1. 基本神经网络模块:")
    
    # 线性层
    print("\n线性层:")
    linear = nn.Linear(10, 5)
    input_tensor = torch.randn(32, 10)
    output = linear(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
    
    # 激活函数
    print("\n激活函数:")
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    
    input_tensor = torch.randn(32, 5)
    relu_output = relu(input_tensor)
    sigmoid_output = sigmoid(input_tensor)
    tanh_output = tanh(input_tensor)
    
    print(f"ReLU输出形状: {relu_output.shape}")
    print(f"Sigmoid输出形状: {sigmoid_output.shape}")
    print(f"Tanh输出形状: {tanh_output.shape}")
    
    # 损失函数
    print("\n损失函数:")
    criterion = nn.MSELoss()
    target = torch.randn(32, 5)
    loss = criterion(output, target)
    print(f"MSE损失: {loss.item()}")

# 2. 自定义神经网络
def custom_neural_network():
    print("\n2. 自定义神经网络:")
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    print("模型结构:")
    print(model)
    
    # 前向传播
    input_tensor = torch.randn(32, 10)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 3. 模型参数
def model_parameters():
    print("\n3. 模型参数:")
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    
    # 查看参数
    print("模型参数:")
    for name, param in model.named_parameters():
        print(f"参数名: {name}, 形状: {param.shape}")
    
    # 访问特定参数
    print("\n访问特定参数:")
    print(f"fc1权重: {model.fc1.weight.shape}")
    print(f"fc1偏置: {model.fc1.bias.shape}")

# 4. 模型训练
def model_training():
    print("\n4. 模型训练:")
    
    # 生成数据
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 5)
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 100
    losses = []
    
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.title('模型训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'model_training_loss.png'))
    plt.show()

# 5. 模型保存和加载
def model_saving_loading():
    print("\n5. 模型保存和加载:")
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    
    # 保存模型
    model_path = os.path.join(script_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型保存到: {model_path}")
    
    # 加载模型
    loaded_model = NeuralNetwork()
    loaded_model.load_state_dict(torch.load(model_path))
    print("模型加载成功")
    
    # 验证加载的模型
    input_tensor = torch.randn(32, 10)
    output1 = model(input_tensor)
    output2 = loaded_model(input_tensor)
    print(f"原始模型输出形状: {output1.shape}")
    print(f"加载模型输出形状: {output2.shape}")
    print(f"输出是否相同: {torch.allclose(output1, output2)}")

# 6. 预训练模型
def pretrained_models():
    print("\n6. 预训练模型:")
    
    # 尝试加载预训练模型（如果可用）
    try:
        from torchvision import models
        
        # 加载预训练的ResNet18
        resnet18 = models.resnet18(pretrained=True)
        print("ResNet18模型加载成功")
        print(f"模型结构: {resnet18}")
        
        # 测试模型
        input_tensor = torch.randn(1, 3, 224, 224)
        output = resnet18(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
    except ImportError:
        print("torchvision 未安装，跳过预训练模型演示")

# 7. 模型评估
def model_evaluation():
    print("\n7. 模型评估:")
    
    # 生成数据
    X = torch.randn(100, 10)
    y = torch.randn(100, 5)
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    
    # 评估模式
    model.eval()
    
    # 前向传播（禁用梯度计算）
    with torch.no_grad():
        outputs = model(X)
        loss = nn.MSELoss()(outputs, y)
        print(f"评估损失: {loss.item():.4f}")
        print(f"输出形状: {outputs.shape}")

# 8. 模型微调
def model_finetuning():
    print("\n8. 模型微调:")
    
    # 定义模型
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(20, 5)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 创建模型
    model = NeuralNetwork()
    
    # 冻结第一层参数
    for param in model.fc1.parameters():
        param.requires_grad = False
    
    # 查看参数状态
    print("参数状态:")
    for name, param in model.named_parameters():
        print(f"参数名: {name}, 需要梯度: {param.requires_grad}")
    
    # 定义优化器（只优化未冻结的参数）
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    print("优化器创建成功")

# 9. 多输入输出模型
def multi_input_output_model():
    print("\n9. 多输入输出模型:")
    
    class MultiInputOutputModel(nn.Module):
        def __init__(self):
            super(MultiInputOutputModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.fc3 = nn.Linear(20, 3)
        
        def forward(self, x1, x2):
            # 合并输入
            x = torch.cat([x1, x2], dim=1)
            x = self.fc1(x)
            x = torch.relu(x)
            # 多个输出
            output1 = self.fc2(x)
            output2 = self.fc3(x)
            return output1, output2
    
    # 创建模型
    model = MultiInputOutputModel()
    print("模型结构:")
    print(model)
    
    # 测试模型
    input1 = torch.randn(32, 5)
    input2 = torch.randn(32, 5)
    output1, output2 = model(input1, input2)
    print(f"输入1形状: {input1.shape}")
    print(f"输入2形状: {input2.shape}")
    print(f"输出1形状: {output1.shape}")
    print(f"输出2形状: {output2.shape}")

# 10. 模型并行
def model_parallel():
    print("\n10. 模型并行:")
    
    # 检查是否有GPU
    if torch.cuda.is_available():
        print("有可用的GPU")
        
        # 定义模型
        class LargeModel(nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                self.fc1 = nn.Linear(1000, 5000)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(5000, 1000)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        
        # 创建模型并移动到GPU
        model = LargeModel().cuda()
        print("模型移动到GPU成功")
        
        # 测试模型
        input_tensor = torch.randn(32, 1000).cuda()
        output = model(input_tensor)
        print(f"输入形状: {input_tensor.shape}")
        print(f"输出形状: {output.shape}")
    else:
        print("没有可用的GPU，跳过模型并行演示")

if __name__ == "__main__":
    basic_nn_modules()
    custom_neural_network()
    model_parameters()
    model_training()
    model_saving_loading()
    pretrained_models()
    model_evaluation()
    model_finetuning()
    multi_input_output_model()
    model_parallel()
    
    print("\n" + "=" * 50)
    print("演示完成！")