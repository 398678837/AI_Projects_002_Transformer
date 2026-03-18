"""
循环神经网络演示
RNN基础与序列处理
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("循环神经网络演示")
print("=" * 70)

# 1. RNN类
print("\n1. 循环神经网络定义...")

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        scale = 0.1
        self.Wh = np.random.randn(hidden_size, hidden_size) * scale
        self.Wx = np.random.randn(input_size, hidden_size) * scale
        self.Wy = np.random.randn(hidden_size, output_size) * scale
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def forward(self, x_sequence):
        self.x_sequence = x_sequence
        self.hidden_states = []
        
        h = np.zeros((1, self.hidden_size))
        self.hidden_states.append(h)
        
        for x in x_sequence:
            h = self.tanh(np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.bh)
            self.hidden_states.append(h)
        
        y = np.dot(h, self.Wy) + self.by
        return y
    
    def backward(self, x_sequence, y_true, learning_rate=0.01):
        seq_len = len(x_sequence)
        
        dy = y_true - self.dy if hasattr(self, 'dy') else y_true - self.forward(x_sequence)
        
        dWy = np.dot(self.hidden_states[-1].T, dy)
        dby = dy.mean(axis=0)
        
        dWh = np.zeros_like(self.Wh)
        dWx = np.zeros_like(self.Wx)
        dbh = np.zeros_like(self.bh)
        
        dh_next = np.dot(dy, self.Wy.T)
        
        for t in reversed(range(seq_len)):
            x = x_sequence[t].reshape(1, -1)
            h_prev = self.hidden_states[t]
            
            dh = dh_next * self.tanh_derivative(h_prev)
            
            dWx += np.dot(x.T, dh)
            dWh += np.dot(h_prev.T, dh)
            dbh += dh.mean(axis=0)
            
            dh_next = np.dot(dh, self.Wh.T)
        
        self.Wh -= learning_rate * dWh
        self.Wx -= learning_rate * dWx
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

# 2. 序列预测示例
print("\n2. 序列预测示例...")

X = np.array([[[1], [2], [3], [4]], 
              [[2], [3], [4], [5]],
              [[3], [4], [5], [6]]])
y = np.array([[[5]], [[6]], [[7]]])

rnn = SimpleRNN(input_size=1, hidden_size=8, output_size=1)

print("训练前预测:")
for seq in X[:1]:
    pred = rnn.forward(seq)
    print(f"  Input: {seq.flatten()}, 预测: {pred[0,0]:.2f}, 真实: {y[0,0,0]}")

print("\n训练中...")
for epoch in range(500):
    for seq, target in zip(X, y):
        rnn.forward(seq)
        rnn.backward(seq, target, learning_rate=0.1)

print("\n训练后预测:")
for seq, target in zip(X, y):
    pred = rnn.forward(seq)
    print(f"  Input: {seq.flatten()}, 预测: {pred[0,0]:.2f}, 真实: {target[0,0]}")

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
inputs = [str(seq.flatten()) for seq in X]
predictions = [rnn.forward(seq)[0,0] for seq in X]
actuals = [target[0,0] for target in y]

x = np.arange(len(inputs))
width = 0.35
ax.bar(x - width/2, predictions, width, label='预测', color='steelblue', alpha=0.7)
ax.bar(x + width/2, actuals, width, label='真实', color='coral', alpha=0.7)
ax.set_xlabel('序列')
ax.set_ylabel('输出值')
ax.set_title('RNN序列Predictions')
ax.set_xticks(x)
ax.set_xticklabels([f'{i}' for i in range(len(inputs))])
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
rnn_structure = """
Input → [RNN Cell] → 输出
        ↓
       隐藏状态"""
ax.text(0.5, 0.5, rnn_structure, fontsize=14, ha='center', va='center',
        transform=ax.transAxes, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('RNN结构')

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'rnn_demo.png'))
print("可视化已保存为 'images/rnn_demo.png'")

# 4. 总结
print("\n" + "=" * 70)
print("RNN总结")
print("=" * 70)
print("""
关键概念:

1. 循环结构:
   - 隐藏状态传递信息
   - 记住之前的信息

2. 时间展开:
   - 沿时间步展开网络
   - 共享权重

3. 梯度问题:
   - 梯度消失
   - 梯度爆炸

4. 应用场景:
   - 序列生成
   - 机器翻译
   - 语音识别
""")
print("=" * 70)
print("\nRNN Demo完成！")
