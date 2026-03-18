"""
编码器-解码器架构演示
Encoder-Decoder Architecture Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Encoder-Decoder Architecture Demo")
print("=" * 70)

# 1. 架构概念
print("\n1. Encoder-Decoder Architecture Concept...")

print("""
Encoder-Decoder (编码器-解码器):
- 编码器处理输入序列
- 解码器生成输出序列
- 用于序列到序列任务
""")

# 2. 架构类型
print("\n2. Architecture Types...")

architectures = {
    'Seq2Seq': {'encoder': 'RNN', 'decoder': 'RNN', 'attention': 'No'},
    'Transformer': {'encoder': 'Multi-Head', 'decoder': 'Multi-Head+Cross', 'attention': 'Self+Cross'},
    'BERT': {'encoder': 'Transformer', 'decoder': 'None', 'attention': 'Bidirectional'},
    'GPT': {'encoder': 'None', 'decoder': 'Transformer', 'attention': 'Masked'}
}

print("\nArchitecture Comparison:")
print("-" * 60)
print(f"{'Architecture':<15} {'Encoder':<20} {'Decoder':<25}")
print("-" * 60)
for arch, components in architectures.items():
    print(f"{arch:<15} {components['encoder']:<20} {components['decoder']:<25}")
print("-" * 60)

# 3. 可视化
print("\n3. Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 编码器-解码器架构
ax = axes[0, 0]
ax.add_patch(plt.Rectangle((0.1, 0.35), 0.35, 0.25, fill=True, color='steelblue', alpha=0.7))
ax.add_patch(plt.Rectangle((0.55, 0.35), 0.35, 0.25, fill=True, color='coral', alpha=0.7))
ax.add_patch(plt.Rectangle((0.4, 0.1), 0.2, 0.15, fill=True, color='green', alpha=0.7))
ax.arrow(0.45, 0.475, 0.05, 0, head_width=0.02, head_length=0.03, fc='black')
ax.text(0.275, 0.5, 'Encoder', ha='center', fontsize=12)
ax.text(0.725, 0.5, 'Decoder', ha='center', fontsize=12)
ax.text(0.5, 0.175, 'Output', ha='center', fontsize=10)
ax.text(0.5, 0.475, 'Context', ha='center', fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Encoder-Decoder Architecture')

# 3.2 BERT vs GPT对比
ax = axes[0, 1]
models = ['BERT', 'GPT']
types = ['Encoder', 'Decoder']
colors = ['steelblue', 'coral']

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, [1, 0], width, label='Encoder', color='steelblue', alpha=0.7)
bars2 = ax.bar(x + width/2, [0, 1], width, label='Decoder', color='coral', alpha=0.7)

ax.set_ylabel('Has Component')
ax.set_title('BERT vs GPT Architecture')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 1.5)

for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1, 
            'Yes' if bar1.get_height() > 0 else 'No', ha='center', va='bottom')
    ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.1, 
            'Yes' if bar2.get_height() > 0 else 'No', ha='center', va='bottom')

# 3.3 不同任务的性能
ax = axes[1, 0]
tasks = ['Translation', 'Summarization', 'QA', 'Dialogue']
accuracy = [0.88, 0.82, 0.85, 0.78]
bars = ax.bar(tasks, accuracy, color='steelblue', alpha=0.7)
ax.set_ylabel('BLEU/ROUGE Score')
ax.set_title('Performance on Different Tasks')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{bar.get_height():.2f}', ha='center', va='bottom')

# 3.4 架构演化
ax = axes[1, 1]
layers = ['RNN', 'CNN', 'Transformer', 'T5', 'GPT-4']
years = [2014, 2015, 2017, 2019, 2023]
performance = [0.6, 0.65, 0.75, 0.85, 0.9]

ax.plot(years, performance, marker='o', linewidth=2, markersize=8, color='steelblue')
ax.fill_between(years, performance, alpha=0.3, color='steelblue')
ax.set_xlabel('Year')
ax.set_ylabel('Performance')
ax.set_title('Architecture Evolution')
ax.grid(True, alpha=0.3)

for i, (year, perf) in enumerate(zip(years, performance)):
    ax.annotate(layers[i], (year, perf), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'encoder_decoder_arch.png'))
print("Visualization saved to 'images/encoder_decoder_arch.png'")

# 4. 架构特点
print("\n4. Architecture Features...")

print("""
Architecture Features:
1. Encoder-Decoder Structure
   - Encoder processes input sequence
   - Decoder generates output sequence
   - Context vector connects encoder and decoder

2. Attention Mechanism
   - Self-attention in encoder
   - Masked self-attention in decoder
   - Cross-attention connects encoder and decoder

3. Parallel Computation
   - Encoder can be computed in parallel
   - Decoder can be computed in parallel (training)
   - Sequential generation (inference)

4. Applications
   - Machine Translation
   - Text Summarization
   - Question Answering
   - Dialogue Systems
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Encoder-Decoder Applications:
1. Machine Translation:
   - Input: Source language sentence
   - Output: Target language sentence
   - Model: Transformer, RNN Seq2Seq

2. Text Summarization:
   - Input: Long text
   - Output: Summary
   - Model: Transformer, BERT2BERT

3. Question Answering:
   - Input: Question + Context
   - Output: Answer
   - Model: BERT, T5

4. Dialogue Systems:
   - Input: Dialogue history
   - Output: Response
   - Model: GPT, Transformer

5. Code Generation:
   - Input: Natural language description
   - Output: Code
   - Model: GPT, CodeBERT
""")

# 6. 总结
print("\n" + "=" * 70)
print("Encoder-Decoder Architecture Summary")
print("=" * 70)

print("""
Key Concepts:
1. Encoder-Decoder is the core architecture for Seq2Seq tasks
2. Encoder processes input, Decoder generates output
3. Attention mechanism captures alignment between input and output
4. Transformer architecture enables parallel computation
5. Widely used in NLP tasks

Architecture Types:
1. Seq2Seq: RNN encoder + decoder
2. Transformer: Attention-based encoder + decoder
3. BERT: Encoder only (understanding)
4. GPT: Decoder only (generation)

Applications:
1. Machine Translation
2. Text Summarization
3. Question Answering
4. Dialogue Systems
5. Code Generation
""")

print("=" * 70)
print("Encoder-Decoder Architecture Demo completed!")
print("=" * 70)
