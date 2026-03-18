"""
分词器演示
Tokenization
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Tokenization Demo")
print("=" * 70)

# 1. 分词器概念
print("\n1. Tokenization Concept...")

print("""
Tokenization:
- Convert text to tokens
- Word / Subword / Character
- Common: BPE, WordPiece, SentencePiece
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = ['BPE', 'WordPiece', 'SentencePiece', 'Character']
vocab_sizes = [30000, 30000, 32000, 50000]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(methods, vocab_sizes, color=colors, alpha=0.7)
ax.set_ylabel('Vocabulary Size', fontsize=10)
ax.set_title('Vocabulary Size by Method', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, size in zip(bars, vocab_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
            f'{size:,}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
text = "Deep learning is a branch of machine learning"
words = text.split()
tokens = ['deep', 'learn', '##ing', 'is', 'a', 'branch', 'of', 'machine', 'learn', '##ing']
x = np.arange(len(tokens))
ax.bar(x, [1]*len(tokens), color='coral', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Token', fontsize=10)
ax.set_title('Tokenization Result', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'tokenization.png'))
print("Visualization saved to 'images/tokenization.png'")

# 3. 分词器特点
print("\n3. Tokenization Features...")

print("""
Tokenization Features:
1. Text Processing
   - Convert text to tokens
   - Handle unknown words
   - Support multiple languages

2. Algorithm Types
   - Word-level
   - Subword-level
   - Character-level

3. Common Algorithms
   - BPE (Byte Pair Encoding)
   - WordPiece
   - SentencePiece

4. Impact on Models
   - Affects model performance
   - Affects computational efficiency
   - Affects model size
""")

# 4. 应用场景
print("\n4. Applications...")

print("""
Tokenization Applications:
1. Text Classification:
   - Sentiment analysis
   - Topic classification
   - Spam detection

2. Named Entity Recognition:
   - Person name recognition
   - Location name recognition
   - Organization recognition

3. Question Answering:
   - Open-domain QA
   - Machine reading comprehension
   - Dialogue systems

4. Text Generation:
   - Text continuation
   - Text summarization
   - Machine translation

5. Code Generation:
   - Code completion
   - Code generation
   - Code translation
""")

# 5. 总结
print("\n" + "=" * 70)
print("Tokenization Summary")
print("=" * 70)

print("""
Key Concepts:
1. Tokenization is the first step in NLP processing
2. Convert text to tokens for model processing
3. Common methods: Word, Subword, Character
4. Common algorithms: BPE, WordPiece, SentencePiece

Tokenization Features:
- Text processing
- Algorithm types
- Common algorithms
- Impact on models

Tokenization Applications:
- Text classification
- Named entity recognition
- Question answering
- Text generation
- Code generation
""")

print("=" * 70)
print("Tokenization Demo completed!")
print("=" * 70)
