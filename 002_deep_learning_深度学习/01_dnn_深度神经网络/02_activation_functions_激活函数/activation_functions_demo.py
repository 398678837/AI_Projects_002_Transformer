"""
激活函数演示
展示各种激活函数的特性
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("激活函数演示")
print("=" * 70)

# 1. 定义各种激活函数
print("\n1. 激活函数定义...")

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 2. 可视化
print("\n2. 可视化激活函数...")

x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

ax = axes[0, 0]
ax.plot(x, sigmoid(x), 'b-', label='Sigmoid', linewidth=2)
ax.plot(x, sigmoid_derivative(x), 'r--', label='导数', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Sigmoid')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(x, tanh(x), 'b-', label='Tanh', linewidth=2)
ax.plot(x, tanh_derivative(x), 'r--', label='导数', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Tanh')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(x, relu(x), 'b-', label='ReLU', linewidth=2)
ax.plot(x, relu_derivative(x), 'r--', label='导数', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('ReLU')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(x, leaky_relu(x), 'b-', label='Leaky ReLU', linewidth=2)
ax.plot(x, leaky_relu_derivative(x), 'r--', label='导数', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Leaky ReLU')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(x, elu(x), 'b-', label='ELU', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('ELU')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
x_softmax = np.array([[1, 2, 3], [2, 4, 1], [1, 1, 1]])
y_softmax = softmax(x_softmax)
ax.bar(range(3), y_softmax[0], color='steelblue', alpha=0.7, label='Input[1,2,3]')
ax.bar(range(3), y_softmax[1], color='coral', alpha=0.7, label='Input[2,4,1]')
ax.set_title('Softmax')
ax.set_xlabel('Class')
ax.set_ylabel('概率')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/activation_functions.png')
print("可视化已保存为 'images/activation_functions.png'")

# 3. 总结表
print("\n3. 激活函数对比...")

summary = """
| 激活函数 | 输出范围 | 优点 | 缺点 |
|----------|----------|------|------|
| Sigmoid | (0,1) | 概率解释 | 梯度消失 |
| Tanh | (-1,1) | 零中心 | 梯度消失 |
| ReLU | [0,∞) | 高效 | Dying ReLU |
| Leaky ReLU | (-∞,∞) | 解决 Dying | 不常用 |
| Softmax | (0,1) | 多分类 | 多类输出 |
"""
print(summary)

print("\n" + "=" * 70)
print("Activation Functions Demo完成！")
print("=" * 70)
