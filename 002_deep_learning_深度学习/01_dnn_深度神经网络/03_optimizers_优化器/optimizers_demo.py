"""
优化器演示
展示各种优化算法的对比
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("优化器演示")
print("=" * 70)

# 1. 定义目标函数
print("\n1. 定义优化目标...")

def objective(x, y):
    return x**2 + 2*y**2

def gradient(x, y):
    return np.array([2*x, 4*y])

# 2. 定义各种优化器
print("\n2. 定义优化器...")

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        return params - self.lr * grads

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.momentum * self.v + self.lr * grads
        return params - self.v

class AdaGrad:
    def __init__(self, lr=0.1, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache += grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)

class RMSprop:
    def __init__(self, lr=0.01, decay=0.9, epsilon=1e-8):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
    
    def update(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)
        self.cache = self.decay * self.cache + (1 - self.decay) * grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# 3. 运行优化
print("\n3. 运行优化算法...")

initial_params = np.array([-4.0, 3.0])
optimizers = {
    'SGD': SGD(lr=0.1),
    'Momentum': Momentum(lr=0.1, momentum=0.9),
    'AdaGrad': AdaGrad(lr=1.0),
    'RMSprop': RMSprop(lr=0.1),
    'Adam': Adam(lr=0.2)
}

history = {}

for name, opt in optimizers.items():
    params = initial_params.copy()
    path = [params.copy()]
    
    for i in range(50):
        grads = gradient(params[0], params[1])
        params = opt.update(params, grads)
        path.append(params.copy())
    
    history[name] = np.array(path)
    final_loss = objective(path[-1][0], path[-1][1])
    print(f"  {name}: 最终损失 = {final_loss:.6f}")

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = objective(X, Y)
ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for (name, path), color in zip(history.items(), colors):
    ax.plot(path[:, 0], path[:, 1], 'o-', label=name, color=color, markersize=3)

ax.scatter(*initial_params, color='black', s=100, marker='*', label='起点')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('优化路径对比')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for (name, path), color in zip(history.items(), colors):
    losses = [objective(p[0], p[1]) for p in path]
    ax.plot(losses, label=name, color=color, linewidth=2)

ax.set_xlabel('迭代次数')
ax.set_ylabel('损失')
ax.set_title('损失曲线对比')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'optimizers.png'))
print("可视化已保存为 'images/optimizers.png'")

# 5. 总结
print("\n" + "=" * 70)
print("优化器总结")
print("=" * 70)
print("""
| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD | 简单直接 | 入门 |
| Momentum | 惯性 | 收敛加速 |
| AdaGrad | 自适应学习率 | 稀疏特征 |
| RMSprop | 指数衰减 | RNN |
| Adam | 综合最优 | 默认首选 |

建议:
- 默认使用Adam
- 追求速度可用SGD+Momentum
""")
print("=" * 70)
print("\nOptimizers Demo完成！")
