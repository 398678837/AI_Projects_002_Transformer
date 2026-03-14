"""
正则化演示
展示Dropout、L1、L2正则化
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("正则化演示")
print("=" * 70)

# 1. 正则化方法
print("\n1. 正则化方法介绍...")

class RegularizedNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.0, l2_lambda=0.0):
        self.weights = []
        self.biases = []
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def dropout(self, x):
        mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
        return x * mask / (1 - self.dropout_rate)
    
    def forward(self, X, training=True):
        self.activations = [X]
        self.dropout_masks = []
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            
            if i < len(self.weights) - 1:
                a = self.relu(z)
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                    self.dropout_masks.append(mask)
                    a = a * mask / (1 - self.dropout_rate)
            else:
                a = self.sigmoid(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        deltas = [None] * len(self.weights)
        
        error = y - self.activations[-1]
        deltas[-1] = error * self.activations[-1] * (1 - self.activations[-1])
        
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[i+1].dot(self.weights[i+1].T)
            deltas[i] = error * (self.activations[i+1] > 0).astype(float)
        
        for i in range(len(self.weights)):
            grad_w = self.activations[i].T.dot(deltas[i]) / m
            grad_b = deltas[i].mean(axis=0)
            
            if self.l2_lambda > 0:
                grad_w += self.l2_lambda * self.weights[i]
            
            self.weights[i] += learning_rate * grad_w
            self.biases[i] += learning_rate * grad_b
    
    def l2_loss(self):
        return self.l2_lambda * sum(np.sum(w**2) for w in self.weights)
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X, training=True)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2) + self.l2_loss()
                losses.append(loss)
                print(f"  Epoch {epoch}: Loss = {loss:.6f}")
        
        return losses

# 2. 对比实验
print("\n2. 对比实验...")

X = np.random.rand(100, 10)
true_weights = np.random.randn(10, 1)
y = X @ true_weights + np.random.randn(100, 1) * 0.1

print("\n无正则化:")
nn_no_reg = RegularizedNetwork([10, 20, 20, 1], dropout_rate=0.0, l2_lambda=0.0)
losses_no_reg = nn_no_reg.train(X, y, epochs=500, learning_rate=0.01)

print("\nL2正则化 (lambda=0.1):")
nn_l2 = RegularizedNetwork([10, 20, 20, 1], dropout_rate=0.0, l2_lambda=0.1)
losses_l2 = nn_l2.train(X, y, epochs=500, learning_rate=0.01)

print("\nDropout (rate=0.3):")
nn_dropout = RegularizedNetwork([10, 20, 20, 1], dropout_rate=0.3, l2_lambda=0.0)
losses_dropout = nn_dropout.train(X, y, epochs=500, learning_rate=0.01)

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(losses_no_reg, label='无正则化', linewidth=2)
ax.plot(losses_l2, label='L2正则化', linewidth=2)
ax.plot(losses_dropout, label='Dropout', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('正则化对比')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
methods = ['无正则化', 'L2', 'Dropout']
weights_norm = [
    np.sum(nn_no_reg.weights[0]**2),
    np.sum(nn_l2.weights[0]**2),
    np.sum(nn_dropout.weights[0]**2)
]
ax.bar(methods, weights_norm, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax.set_ylabel('权重范数')
ax.set_title('权重复杂度对比')
ax.grid(True, alpha=0.3)

for i, v in enumerate(weights_norm):
    ax.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('images/regularization.png')
print("可视化已保存为 'images/regularization.png'")

# 4. 总结
print("\n" + "=" * 70)
print("正则化总结")
print("=" * 70)
print("""
| 方法 | 作用 | 适用场景 |
|------|------|----------|
| L1 | 稀疏化 | 特征选择 |
| L2 | 权重衰减 | 防止过拟合 |
| Dropout | 随机失活 | 深度网络 |
| Early Stopping | 早停 | 通用 |

正则化是防止过拟合的重要技术。
""")
print("=" * 70)
print("\nRegularization Demo完成！")
