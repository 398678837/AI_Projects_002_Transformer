"""
LSTM演示
长短期记忆网络
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LSTM演示")
print("=" * 70)

# 1. LSTM概念
print("\n1. LSTM概念...")

print("""
LSTM (Long Short-Term Memory):
- 解决长序列梯度问题
- 引入门控机制
- 选择性记住/忘记信息

核心组件:
1. 遗忘门: 决定丢弃什么信息
2. 输入门: 决定存储什么信息  
3. 输出门: 决定输出什么信息
""")

# 2. LSTM单元
print("\n2. LSTM单元结构...")

class SimpleLSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = 0.1
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * scale
        
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x_sequence):
        self.x_sequence = x_sequence
        T = len(x_sequence)
        
        h = np.zeros((1, self.hidden_size))
        c = np.zeros((1, self.hidden_size))
        
        self.hs = [h]
        self.cs = [c]
        
        for t in range(T):
            x = x_sequence[t].reshape(1, -1)
            combined = np.concatenate([h, x], axis=1)
            
            f = self.sigmoid(np.dot(combined, self.Wf) + self.bf)
            i = self.sigmoid(np.dot(combined, self.Wi) + self.bi)
            c_tilde = self.tanh(np.dot(combined, self.Wc) + self.bc)
            
            c = f * c + i * c_tilde
            
            o = self.sigmoid(np.dot(combined, self.Wo) + self.bo)
            h = o * self.tanh(c)
            
            self.hs.append(h)
            self.cs.append(c)
        
        return h

# 3. 序列预测
print("\n3. 序列预测示例...")

np.random.seed(42)
X = [np.random.randn(1, 1) for _ in range(5)]

lstm = SimpleLSTM(input_size=1, hidden_size=4)

output = lstm.forward(X)
print(f"  序列长度: {len(X)}")
print(f"  隐藏层大小: {lstm.hidden_size}")
print(f"  输出: {output}")

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.add_patch(plt.Rectangle((0.2, 0.3), 0.6, 0.4, facecolor='lightblue', edgecolor='black', linewidth=2))
ax.text(0.5, 0.5, 'LSTM Cell', ha='center', va='center', fontsize=14)

ax.annotate('', xy=(0.8, 0.5), xytext=(1.0, 0.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.9, 0.6, 'h_t', fontsize=12, color='green')

ax.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.1),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(0.55, 0.2, 'c_t', fontsize=12, color='blue')

ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('LSTM单元', fontsize=14)

ax = axes[1]
gates = ['遗忘门\n(Forget)', '输入门\n(Input)', '候选值\n(Candidate)', '输出门\n(Output)']
importance = [0.85, 0.75, 0.65, 0.80]
colors = ['steelblue', 'coral', 'green', 'purple']

bars = ax.bar(gates, importance, color=colors, alpha=0.7)
ax.set_ylabel('重要性')
ax.set_title('LSTM门控机制')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

for bar, imp in zip(bars, importance):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
           f'{imp:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('images/lstm_demo.png')
print("可视化已保存为 'images/lstm_demo.png'")

# 5. 总结
print("\n" + "=" * 70)
print("LSTM总结")
print("=" * 70)
print("""
| 门 | 作用 | 公式 |
|-----|------|------|
| 遗忘门 | 丢弃信息 | f = σ(W_f·[h_{t-1},x_t]) |
| 输入门 | 添加新信息 | i = σ(W_i·[h_{t-1},x_t]) |
| 候选值 | 候选记忆 | C̃ = tanh(W_C·[h_{t-1},x_t]) |
| 输出门 | 决定输出 | o = σ(W_o·[h_{t-1},x_t]) |

优点:
- 解决长序列梯度消失
- 长期依赖建模
- 灵活的门控机制
""")
print("=" * 70)
print("\nLSTM Demo完成！")
