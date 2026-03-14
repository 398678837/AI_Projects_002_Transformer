"""
分词器演示
Tokenization
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("分词器演示")
print("=" * 70)

# 1. 分词器概念
print("\n1. 分词器概念...")

print("""
分词器 (Tokenization):
- 将文本转换为token
- Word / Subword / Character
- 常用: BPE, WordPiece, SentencePiece
""")

# 2. 可视化
print("\n2. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
methods = ['BPE', 'WordPiece', 'SentencePiece', 'Character']
vocab_sizes = [30000, 30000, 32000, 50000]
ax.bar(methods, vocab_sizes, color='steelblue', alpha=0.7)
ax.set_ylabel('词表大小')
ax.set_title('不同分词器词表大小')
ax.grid(True, alpha=0.3)

ax = axes[1]
text = "深度学习是机器学习的分支"
tokens = ['深', '度', '学习', '是', '机', '器', '学', '习', '的', '分', '支']
ax.barh(range(len(tokens)), [1]*len(tokens), color='coral', alpha=0.7)
ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens)
ax.set_xlabel('Token')
ax.set_title('分词结果')

plt.tight_layout()
plt.savefig('images/tokenization.png')
print("可视化已保存为 'images/tokenization.png'")

print("\n" + "=" * 70)
print("分词器总结")
print("=" * 70)
print("分词器将文本转换为模型可处理的token。")
print("=" * 70)
