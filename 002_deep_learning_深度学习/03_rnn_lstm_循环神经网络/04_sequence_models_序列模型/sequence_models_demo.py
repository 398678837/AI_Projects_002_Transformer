"""
序列模型应用演示
文本生成、时间序列、机器翻译
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("序列模型应用演示")
print("=" * 70)

# 1. 文本生成
print("\n1. 文本生成 (Text Generation)")
print("-" * 50)
print("""
应用:
- 诗歌创作
- 代码生成
- 对话系统

技术:
- 字符级/词级RNN
- 温度采样
- Beam Search
""")

# 2. 时间序列预测
print("\n2. 时间序列预测 (Time Series)")
print("-" * 50)
print("""
应用:
- 股票预测
- 天气预测
- 能源消耗

技术:
- 多步预测
- 滑动窗口
- 注意力机制
""")

# 3. 机器翻译
print("\n3. 机器翻译 (Machine Translation)")
print("-" * 50)
print("""
应用:
- 语言互译
- 文档翻译
- 实时翻译

技术:
- Encoder-Decoder
- 注意力机制
- Transformer
""")

# 4. 可视化对比
print("\n4. 应用对比可视化...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

applications = ['文本生成', '时间序列', '机器翻译']
accuracy = [0.72, 0.65, 0.85]
difficulty = [8, 7, 9]
popularity = [9, 7, 8]

ax = axes[0]
bars = ax.bar(applications, accuracy, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax.set_ylabel('Accuracy')
ax.set_title('应用Accuracy对比')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
for bar, acc in zip(bars, accuracy):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
           f'{acc:.0%}', ha='center', fontsize=10)

ax = axes[1]
bars = ax.bar(applications, difficulty, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax.set_ylabel('难度 (1-10)')
ax.set_title('应用难度对比')
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

ax = axes[2]
bars = ax.bar(applications, popularity, color=['steelblue', 'coral', 'green'], alpha=0.7)
ax.set_ylabel('流行度 (1-10)')
ax.set_title('应用流行度对比')
ax.set_ylim(0, 10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'sequence_applications.png'))
print("可视化已保存为 'images/sequence_applications.png'")

# 5. 序列模型架构对比
print("\n5. 架构对比...")

fig, ax = plt.subplots(figsize=(10, 6))

architectures = [
    'RNN\n(基础)',
    'LSTM\n(长依赖)',
    'GRU\n(轻量)',
    'Seq2Seq\n(编码-解码)',
    'Transformer\n(注意力)'
]

performance = [0.65, 0.78, 0.76, 0.82, 0.92]
complexity = [3, 5, 4, 7, 9]

x = np.arange(len(architectures))
width = 0.35

ax.bar(x - width/2, performance, width, label='性能', color='steelblue', alpha=0.7)
ax.bar(x + width/2, [c/1.5 for c in complexity], width, label='复杂度(归一化)', color='coral', alpha=0.7)
ax.set_ylabel('分数')
ax.set_title('序列模型架构对比')
ax.set_xticks(x)
ax.set_xticklabels(architectures)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'sequence_models.png'))
print("可视化已保存为 'images/sequence_models.png'")

# 6. 总结
print("\n" + "=" * 70)
print("序列模型总结")
print("=" * 70)
print("""
| 应用 | 典型模型 | 特点 |
|------|----------|------|
| 文本生成 | CharRNN, GPT | 自回归 |
| 时间序列 | LSTM, Transformer | 多步预测 |
| 机器翻译 | Seq2Seq, Transformer | Encoder-Decoder |

现代主流:
- Transformer成为序列建模标配
- 预训练模型 (BERT, GPT) 广泛应用
""")
print("=" * 70)
print("\nSequence Models Demo完成！")
