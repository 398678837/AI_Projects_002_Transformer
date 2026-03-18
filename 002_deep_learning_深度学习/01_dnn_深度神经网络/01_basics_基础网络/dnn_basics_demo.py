"""
DNN基础网络演示
多层感知机与反向传播
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("DNN基础网络演示")
print("=" * 70)

# 1. 简单的神经网络类
print("\n1. 神经网络定义...")

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        self.layer_sizes = layer_sizes
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        error = y - self.activations[-1]
        deltas[-1] = error * self.sigmoid_derivative(self.activations[-1])
        
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1].dot(self.weights[i+1].T)
            deltas[i] = error * self.relu_derivative(self.activations[i+1])
        
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.activations[i].T.dot(deltas[i]) / m
            self.biases[i] += learning_rate * deltas[i].mean(axis=0)
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                losses.append(loss)
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        return losses

# 2. 训练XOR问题
print("\n2. 训练神经网络解决XOR问题...")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 8, 8, 1])
losses = nn.train(X, y, epochs=2000, learning_rate=0.1)

print("\nPredictions:")
for x in X:
    pred = nn.forward(x.reshape(1, -1))[0, 0]
    print(f"  {x} -> {pred:.4f}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
epochs_range = range(0, len(losses) * 100, 100)
ax.plot(epochs_range, losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Curve')
ax.grid(True, alpha=0.3)

ax = axes[1]
predictions = []
for x in X:
    pred = nn.forward(x.reshape(1, -1))[0, 0]
    predictions.append(pred)

labels = [f'{x[0]},{x[1]}' for x in X]
ax.bar(labels, predictions, color='steelblue', alpha=0.7)
ax.axhline(y=0.5, color='red', linestyle='--', label='Decision Boundary')
ax.set_xlabel('Input')
ax.set_ylabel('Prediction Probability')
ax.set_title('XOR Problem Predictions')
ax.legend()
ax.grid(True, alpha=0.3)

for i, v in enumerate(predictions):
    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'dnn_basics.png'))
print("可视化已保存为 'images/dnn_basics.png'")

# 4. 总结
print("\n" + "=" * 70)
print("DNN基础网络总结")
print("=" * 70)
print("""
关键概念:

1. 深度神经网络:
   - 多层结构 (Input层、隐藏层、输出层)
   - 非线性激活函数
   - 可以逼近任意复杂函数

2. 反向传播:
   - 计算梯度
   - 链式法则
   - 参数更新

3. XOR问题:
   - 单层感知机无法解决
   - 多层神经网络可以解决
""")
print("=" * 70)
print("\nDNN Basics Demo完成！")
