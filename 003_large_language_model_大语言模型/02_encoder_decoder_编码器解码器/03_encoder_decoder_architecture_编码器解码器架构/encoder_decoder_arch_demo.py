"""
编码器-解码器架构演示
Encoder-Decoder Architecture
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("编码器-解码器架构演示")
print("=" * 70)

# 1. 架构概念
print("\n1. 编码器-解码器架构...")

print("""
编码器-解码器 (Encoder-Decoder):
- 编码器处理输入
- 解码器生成输出
- 用于序列到序列任务
""")

# 2. 架构对比
print("\n2. 不同架构对比...")

models = {
    'Seq2Seq': {'encoder': 'RNN', 'decoder': 'RNN'},
    'Transformer': {'encoder': 'Multi-Head', 'decoder': 'Multi-Head+Cross'},
    'BERT': {'encoder': 'Transformer', 'decoder': '无'},
    'GPT': {'encoder': '无', 'decoder': 'Transformer'}
}

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.add_patch(plt.Rectangle((0.1, 0.4), 0.35, 0.2, fill=True, color='steelblue', alpha=0.7))
ax.add_patch(plt.Rectangle((0.55, 0.4), 0.35, 0.2, fill=True, color='coral', alpha=0.7))
ax.add_patch(plt.Rectangle((0.4, 0.15), 0.2, 0.15, fill=True, color='green', alpha=0.7))
ax.arrow(0.45, 0.5, 0.05, -0.1, head_width=0.03, head_length=0.03, fc='black')
ax.text(0.275, 0.5, 'Encoder', ha='center', fontsize=12)
ax.text(0.725, 0.5, 'Decoder', ha='center', fontsize=12)
ax.text(0.5, 0.22, '输出', ha='center', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Encoder-Decoder架构')

ax = axes[1]
tasks = ['翻译', '摘要', '问答', '对话']
accuracy = [0.88, 0.82, 0.85, 0.78]
ax.bar(tasks, accuracy, color='steelblue', alpha=0.7)
ax.set_ylabel('准确率')
ax.set_title('不同任务的性能')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/encoder_decoder_arch.png')
print("可视化已保存为 'images/encoder_decoder_arch.png'")

# 4. 总结
print("\n" + "=" * 70)
print("编码器-解码器总结")
print("=" * 70)
print("""
架构类型:
1. Seq2Seq: RNN编码器+解码器
2. Transformer: 注意力机制
3. BERT: 仅编码器
4. GPT: 仅解码器
""")
print("=" * 70)
print("\nEncoder-Decoder Architecture Demo完成！")
