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

print("PyTorch 循环神经网络演示")
print("=" * 50)

# 1. 基本RNN模型
def basic_rnn():
    print("\n1. 基本RNN模型:")
    
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
    
    # 创建模型
    model = SimpleRNN(input_size=10, hidden_size=50, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 10, 10)  # (batch_size, sequence_length, input_size)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 2. LSTM模型
def lstm_model():
    print("\n2. LSTM模型:")
    
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_size)
            c0 = torch.zeros(1, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型
    model = LSTMModel(input_size=10, hidden_size=50, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 10, 10)  # (batch_size, sequence_length, input_size)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 3. GRU模型
def gru_model():
    print("\n3. GRU模型:")
    
    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_size)
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型
    model = GRUModel(input_size=10, hidden_size=50, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 10, 10)  # (batch_size, sequence_length, input_size)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 4. RNN训练
def rnn_training():
    print("\n4. RNN训练:")
    
    # 生成序列数据
    def generate_sequence_data(n_samples, sequence_length, input_size):
        X = torch.randn(n_samples, sequence_length, input_size)
        y = torch.sum(X, dim=1)
        return X, y
    
    # 生成数据
    X, y = generate_sequence_data(n_samples=1000, sequence_length=10, input_size=5)
    
    # 划分训练集和测试集
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    # 定义LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(1, x.size(0), self.hidden_size)
            c0 = torch.zeros(1, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型、损失函数和优化器
    model = LSTMModel(input_size=5, hidden_size=50, output_size=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 50
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_X)
            test_loss = criterion(test_outputs, test_y)
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    # 可视化损失
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('RNN训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(images_dir, 'rnn_training_loss.png'))
    plt.show()

# 5. 双向RNN
def bidirectional_rnn():
    print("\n5. 双向RNN:")
    
    class BidirectionalRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalRNN, self).__init__()
            self.hidden_size = hidden_size
            self.bi_rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(2, x.size(0), self.hidden_size)  # 双向，所以是2
            out, _ = self.bi_rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型
    model = BidirectionalRNN(input_size=10, hidden_size=50, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 10, 10)  # (batch_size, sequence_length, input_size)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 6. 多层RNN
def multilayer_rnn():
    print("\n6. 多层RNN:")
    
    class MultilayerRNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(MultilayerRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型
    model = MultilayerRNN(input_size=10, hidden_size=50, num_layers=2, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试模型
    input_tensor = torch.randn(32, 10, 10)  # (batch_size, sequence_length, input_size)
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")

# 7. RNN文本分类
def rnn_text_classification():
    print("\n7. RNN文本分类:")
    
    # 生成文本数据
    def generate_text_data(n_samples, sequence_length, vocab_size):
        X = torch.randint(0, vocab_size, (n_samples, sequence_length))
        y = torch.randint(0, 2, (n_samples,))
        return X, y
    
    # 生成数据
    vocab_size = 100
    X, y = generate_text_data(n_samples=1000, sequence_length=20, vocab_size=vocab_size)
    
    # 划分训练集和测试集
    train_X, test_X = X[:800], X[800:]
    train_y, test_y = y[:800], y[800:]
    
    # 定义文本分类模型
    class TextClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
            super(TextClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            embedded = self.embedding(x)
            h0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
            c0 = torch.zeros(1, x.size(0), self.lstm.hidden_size)
            out, _ = self.lstm(embedded, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    # 创建模型、损失函数和优化器
    model = TextClassifier(vocab_size=vocab_size, embedding_dim=50, hidden_size=100, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    num_epochs = 20
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_X)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == test_y).sum().item() / test_y.size(0)
            test_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    ax1.plot(train_losses)
    ax1.set_title('训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(test_accuracies)
    ax2.set_title('测试准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'rnn_text_classification.png'))
    plt.show()

if __name__ == "__main__":
    basic_rnn()
    lstm_model()
    gru_model()
    rnn_training()
    bidirectional_rnn()
    multilayer_rnn()
    rnn_text_classification()
    
    print("\n" + "=" * 50)
    print("演示完成！")