"""
GRU演示
门控循环单元
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("GRU演示")
print("=" * 70)

# 1. GRU概念
print("\n1. GRU概念...")

print("""
GRU (Gated Recurrent Unit):
- LSTM的简化版本
- 参数更少
- 性能相当

核心组件:
1. 更新门: 控制过去信息保留多少
2. 重置门: 控制忽略多少过去信息
""")

# 2. GRU结构
print("\n2. GRU结构...")

class SimpleGRU:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = 0.1
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * scale
        
        self.bz = np.zeros((1, hidden_size))
        self.br = np.zeros((1, hidden_size))
        self.bh = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_sequence):
        self.x_sequence = x_sequence
        h = np.zeros((1, self.hidden_size))
        
        for x in x_sequence:
            x = x.reshape(1, -1)
            combined = np.concatenate([h, x], axis=1)
            
            z = self.sigmoid(np.dot(combined, self.Wz) + self.bz)
            r = self.sigmoid(np.dot(combined, self.Wr) + self.br)
            
            combined_r = np.concatenate([r * h, x], axis=1)
            h_tilde = self.tanh(np.dot(combined_r, self.Wh) + self.bh)
            
            h = (1 - z) * h + z * h_tilde
        
        return h

# 3. 示例
print("\n3. GRU示例...")

np.random.seed(42)
X = [np.random.randn(1, 1) for _ in range(5)]

gru = SimpleGRU(input_size=1, hidden_size=4)

output = gru.forward(X)
print(f"  序列长度: {len(X)}")
print(f"  隐藏层大小: {gru.hidden_size}")
print(f"  输出: {output}")

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
components = ['更新门\n(Update)', '重置门\n(Reset)', '候选隐藏状态\n(Candidate)']
importance = [0.82, 0.78, 0.70]
colors = ['steelblue', 'coral', 'green']

bars = ax.bar(components, importance, color=colors, alpha=0.7)
ax.set_ylabel('重要性')
ax.set_title('GRU门控机制')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

ax = axes[1]
models = ['RNN', 'LSTM', 'GRU']
params = [1, 4, 3]
speed = [1.0, 0.7, 0.9]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, params, width, label='参数量', color='steelblue', alpha=0.7)
ax.bar(x + width/2, speed, width, label='计算速度', color='coral', alpha=0.7)
ax.set_ylabel('相对值')
ax.set_title('RNN变体对比')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/gru_demo.png')
print("可视化已保存为 'images/gru_demo.png'")

# 5. 总结
print("\n" + "=" * 70)
print("GRU总结")
print("=" * 70)
print("""
| 组件 | 作用 |
|------|------|
| 更新门 | 控制过去信息保留多少 |
| 重置门 | 控制忽略多少过去信息 |

对比:
- GRU: 3个门, 较少参数
- LSTM: 4个门, 较多参数

GRU参数少，训练快，性能与LSTM相当。
""")
print("=" * 70)
print("\nGRU Demo完成！")
