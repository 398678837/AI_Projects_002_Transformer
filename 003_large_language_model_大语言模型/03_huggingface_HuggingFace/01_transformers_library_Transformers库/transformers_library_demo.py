"""
Transformers库演示
Hugging Face Transformers
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Transformers Library Demo")
print("=" * 70)

# 1. Transformers库概念
print("\n1. Transformers Library Concept...")

print("""
Transformers Library:
- Most popular NLP deep learning library
- Provides pre-trained models
- Simplifies model usage
""")

# 2. 常用模型
print("\n2. Common Models...")

models_info = {
    'BERT': {'type': 'Encoder', 'params': '110M', 'task': 'Understanding'},
    'GPT-2': {'type': 'Decoder', 'params': '124M', 'task': 'Generation'},
    'T5': {'type': 'Encoder-Decoder', 'params': '220M', 'task': 'Seq2Seq'},
    'Llama': {'type': 'Decoder', 'params': '7B', 'task': 'Generation'},
    'Qwen': {'type': 'Decoder', 'params': '1.8B', 'task': 'Chat'}
}

print("\nModel Information:")
for model, info in models_info.items():
    print(f"  {model}: {info['type']}, {info['params']}, {info['task']}")

# 3. 可视化
print("\n3. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 3.1 模型参数量对比
ax = axes[0]
model_names = list(models_info.keys())
param_counts = [float(m['params'].replace('M', '').replace('B', '000')) for m in models_info.values()]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(model_names, param_counts, color=colors, alpha=0.7)
ax.set_ylabel('Parameter Count (Million)', fontsize=10)
ax.set_title('Model Parameter Comparison', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, count in zip(bars, param_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{count:.1f}', ha='center', va='bottom', fontsize=9)

# 3.2 模型使用率对比
ax = axes[1]
usage = [95, 88, 75, 70, 65]
bars = ax.bar(model_names, usage, color=colors, alpha=0.7)
ax.set_ylabel('Usage Rate (%)', fontsize=10)
ax.set_title('Model Usage Rate', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 100)

# 添加数值标签
for bar, use in zip(bars, usage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{use}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'transformers_library.png'))
print("Visualization saved to 'images/transformers_library.png'")

# 4. Transformers库特点
print("\n4. Transformers Library Features...")

print("""
Transformers Library Features:
1. Rich Pre-trained Models
   - 100+ model architectures
   - 10000+ pre-trained models
   - Easy loading interface

2. Simple and Easy-to-use API
   - Pipeline interface
   - AutoModel/AutoTokenizer
   - Consistent design

3. Active Community Support
   - Rich documentation
   - Active community
   - Regular updates

4. Powerful Functionality
   - Support for multiple NLP tasks
   - Model fine-tuning
   - Distributed training

5. Continuous Updates
   - New models added regularly
   - Performance improvements
   - New features
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Transformers Library Applications:
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

# 6. 总结
print("\n" + "=" * 70)
print("Transformers Library Summary")
print("=" * 70)

print("""
Key Concepts:
1. Transformers is the most popular NLP library
2. Provides 100+ model architectures
3. Simple and easy-to-use API
4. Active community support
5. Powerful functionality

Transformers Library Features:
- Rich pre-trained models
- Simple API
- Active community
- Powerful functionality
- Continuous updates

Transformers Library Applications:
- Text classification
- Named entity recognition
- Question answering
- Text generation
- Code generation
""")

print("=" * 70)
print("Transformers Library Demo completed!")
print("=" * 70)
