"""
Activation Functions Demo
Demonstrates various activation functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Activation Functions Demo")
print("=" * 70)

# 1. Define various activation functions
print("\n1. Define activation functions...")

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

# 2. Visualization
print("\n2. Visualize activation functions...")

x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

ax = axes[0, 0]
ax.plot(x, sigmoid(x), 'b-', label='Sigmoid', linewidth=2)
ax.plot(x, sigmoid_derivative(x), 'r--', label='Derivative', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Sigmoid')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(x, tanh(x), 'b-', label='Tanh', linewidth=2)
ax.plot(x, tanh_derivative(x), 'r--', label='Derivative', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Tanh')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(x, relu(x), 'b-', label='ReLU', linewidth=2)
ax.plot(x, relu_derivative(x), 'r--', label='Derivative', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('ReLU')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(x, leaky_relu(x), 'b-', label='Leaky ReLU', linewidth=2)
ax.plot(x, leaky_relu_derivative(x), 'r--', label='Derivative', linewidth=2)
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
ax.set_ylabel('Probability')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'activation_functions.png'))
print("Visualization saved to 'images/activation_functions.png'")

# 3. Summary table
print("\n3. Activation Functions Comparison...")

summary = """
| Activation Function | Output Range | Pros | Cons |
|---------------------|---------------|------|------|
| Sigmoid | (0,1) | Probabilistic output | Gradient vanishing |
| Tanh | (-1,1) | Zero-centered | Gradient vanishing |
| ReLU | [0,∞) | Efficient | Dying ReLU |
| Leaky ReLU | (-∞,∞) | Fixes Dying ReLU | Less commonly used |
| Softmax | (0,1) | Multi-class | Multi-output |
"""
print(summary)

print("\n" + "=" * 70)
print("Activation Functions Demo completed!")
print("=" * 70)
