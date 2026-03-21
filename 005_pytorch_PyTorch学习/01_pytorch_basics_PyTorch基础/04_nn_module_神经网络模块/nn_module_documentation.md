# PyTorch 神经网络模块详细文档

## 1. 神经网络模块的基本概念

PyTorch的`torch.nn`模块提供了构建神经网络所需的各种组件，包括层、激活函数、损失函数等。这些组件都是`nn.Module`的子类，它们可以组合在一起形成复杂的神经网络。

## 2. 基本神经网络模块

### 2.1 线性层

线性层（`nn.Linear`）是最基本的神经网络层，它执行线性变换： y = xA^T + b 。

```python
import torch
import torch.nn as nn

# 创建线性层
linear = nn.Linear(10, 5)  # 输入特征数为10，输出特征数为5

# 输入张量
input_tensor = torch.randn(32, 10)  # 批量大小为32，特征数为10

# 前向传播
output = linear(input_tensor)
print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([32, 10])
print(f"输出形状: {output.shape}")        # 输出: torch.Size([32, 5])
```

### 2.2 激活函数

激活函数为神经网络引入非线性，常见的激活函数包括：

- **ReLU**： f(x) = max(0, x) 
- **Sigmoid**： f(x) = 1 / (1 + e^{-x}) 
- **Tanh**： f(x) = (e^x - e^{-x}) / (e^x + e^{-x}) 

```python
# 激活函数
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

input_tensor = torch.randn(32, 5)
relu_output = relu(input_tensor)
sigmoid_output = sigmoid(input_tensor)
tanh_output = tanh(input_tensor)

print(f"ReLU输出形状: {relu_output.shape}")      # 输出: torch.Size([32, 5])
print(f"Sigmoid输出形状: {sigmoid_output.shape}")  # 输出: torch.Size([32, 5])
print(f"Tanh输出形状: {tanh_output.shape}")        # 输出: torch.Size([32, 5])
```

### 2.3 损失函数

损失函数用于计算模型预测与真实值之间的差异，常见的损失函数包括：

- **MSE Loss**：均方误差损失
- **CrossEntropyLoss**：交叉熵损失
- **BCELoss**：二元交叉熵损失

```python
# 损失函数
criterion = nn.MSELoss()
target = torch.randn(32, 5)
loss = criterion(output, target)
print(f"MSE损失: {loss.item()}")
```

## 3. 自定义神经网络

### 3.1 基本结构

创建自定义神经网络需要继承`nn.Module`类，并实现`__init__`和`forward`方法。

```python
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
print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([32, 10])
print(f"输出形状: {output.shape}")        # 输出: torch.Size([32, 5])
```

### 3.2 模型参数

神经网络的参数包括权重和偏置，它们可以通过`named_parameters()`方法访问。

```python
# 查看参数
print("模型参数:")
for name, param in model.named_parameters():
    print(f"参数名: {name}, 形状: {param.shape}")

# 访问特定参数
print("\n访问特定参数:")
print(f"fc1权重: {model.fc1.weight.shape}")  # 输出: torch.Size([20, 10])
print(f"fc1偏置: {model.fc1.bias.shape}")    # 输出: torch.Size([20])
```

## 4. 模型训练

### 4.1 基本训练流程

模型训练的基本流程包括：
1. 前向传播
2. 计算损失
3. 反向传播
4. 更新参数

```python
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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

## 5. 模型保存和加载

### 5.1 保存模型

可以使用`torch.save`保存模型的状态字典。

```python
# 保存模型
model_path = 'model.pth'
torch.save(model.state_dict(), model_path)
print(f"模型保存到: {model_path}")
```

### 5.2 加载模型

可以使用`torch.load`加载模型的状态字典。

```python
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
```

## 6. 预训练模型

PyTorch提供了许多预训练模型，可以直接使用这些模型进行特征提取或微调。

```python
from torchvision import models

# 加载预训练的ResNet18
resnet18 = models.resnet18(pretrained=True)
print("ResNet18模型加载成功")

# 测试模型
input_tensor = torch.randn(1, 3, 224, 224)
output = resnet18(input_tensor)
print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([1, 3, 224, 224])
print(f"输出形状: {output.shape}")        # 输出: torch.Size([1, 1000])
```

## 7. 模型评估

在评估模型时，应该将模型设置为评估模式，并禁用梯度计算。

```python
# 评估模式
model.eval()

# 前向传播（禁用梯度计算）
with torch.no_grad():
    outputs = model(X)
    loss = nn.MSELoss()(outputs, y)
    print(f"评估损失: {loss.item():.4f}")
    print(f"输出形状: {outputs.shape}")
```

## 8. 模型微调

模型微调是指在预训练模型的基础上，针对特定任务进行训练。

```python
# 冻结第一层参数
for param in model.fc1.parameters():
    param.requires_grad = False

# 查看参数状态
print("参数状态:")
for name, param in model.named_parameters():
    print(f"参数名: {name}, 需要梯度: {param.requires_grad}")

# 定义优化器（只优化未冻结的参数）
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
print("优化器创建成功")
```

## 9. 多输入输出模型

神经网络可以处理多个输入和多个输出。

```python
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

# 测试模型
input1 = torch.randn(32, 5)
input2 = torch.randn(32, 5)
output1, output2 = model(input1, input2)
print(f"输入1形状: {input1.shape}")  # 输出: torch.Size([32, 5])
print(f"输入2形状: {input2.shape}")  # 输出: torch.Size([32, 5])
print(f"输出1形状: {output1.shape}")  # 输出: torch.Size([32, 5])
print(f"输出2形状: {output2.shape}")  # 输出: torch.Size([32, 3])
```

## 10. 模型并行

对于大型模型，可以使用GPU进行并行计算。

```python
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
    print(f"输入形状: {input_tensor.shape}")  # 输出: torch.Size([32, 1000])
    print(f"输出形状: {output.shape}")        # 输出: torch.Size([32, 1000])
else:
    print("没有可用的GPU，跳过模型并行演示")
```

## 11. 常见问题与解决方案

### 11.1 模型参数不更新

**问题**：模型训练时参数不更新

**解决方案**：
- 检查是否调用了`optimizer.step()`
- 检查是否在反向传播前调用了`optimizer.zero_grad()`
- 检查参数的`requires_grad`属性是否为True

### 11.2 过拟合

**问题**：模型在训练集上表现良好，但在测试集上表现差

**解决方案**：
- 添加正则化（如L1/L2正则化）
- 使用dropout
- 增加训练数据
- 简化模型结构

### 11.3 内存不足

**问题**：训练时出现内存不足错误

**解决方案**：
- 减小批量大小
- 使用梯度累积
- 使用混合精度训练
- 模型并行或数据并行

### 11.4 模型加载错误

**问题**：加载模型时出现错误

**解决方案**：
- 确保模型结构与保存时一致
- 确保使用相同版本的PyTorch
- 检查模型路径是否正确

## 12. 高级神经网络模块

### 12.1 卷积神经网络

卷积神经网络（CNN）是一种专门用于处理网格数据的神经网络，如图像。

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        return x
```

### 12.2 循环神经网络

循环神经网络（RNN）是一种专门用于处理序列数据的神经网络。

```python
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
```

## 13. 总结

PyTorch的`nn`模块提供了构建神经网络所需的各种组件，包括层、激活函数、损失函数等。本文档介绍了神经网络模块的基本用法，包括：

- 基本神经网络模块
- 自定义神经网络
- 模型参数
- 模型训练
- 模型保存和加载
- 预训练模型
- 模型评估
- 模型微调
- 多输入输出模型
- 模型并行
- 常见问题与解决方案
- 高级神经网络模块

通过这些内容，您应该已经掌握了PyTorch神经网络模块的基本用法，可以开始构建和训练各种神经网络模型了。