"""
Transformers库演示
Hugging Face Transformers
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Transformers库演示")
print("=" * 70)

# 1. Transformers库概念
print("\n1. Transformers库概念...")

print("""
Hugging Face Transformers:
- 最流行的NLP深度学习库
- 提供预训练模型
- 简化模型使用
""")

# 2. 常用模型
print("\n2. 常用模型...")

models_info = {
    'BERT': {'type': 'Encoder', 'params': '110M', 'task': '理解'},
    'GPT-2': {'type': 'Decoder', 'params': '124M', 'task': '生成'},
    'T5': {'type': 'Encoder-Decoder', 'params': '220M', 'task': 'Seq2Seq'},
    'Llama': {'type': 'Decoder', 'params': '7B', 'task': '生成'},
    'Qwen': {'type': 'Decoder', 'params': '1.8B', 'task': '对话'}
}

# 3. 可视化
print("\n3. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
model_names = list(models_info.keys())
param_counts = [float(m['params'].replace('M', '').replace('B', '000')) for m in models_info.values()]
ax.bar(model_names, param_counts, color='steelblue', alpha=0.7)
ax.set_ylabel('参数量 (M)')
ax.set_title('不同模型参数量')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

ax = axes[1]
usage = [95, 88, 75, 70, 65]
ax.bar(model_names, usage, color='coral', alpha=0.7)
ax.set_ylabel('使用率 (%)')
ax.set_title('模型使用率')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/transformers_library.png')
print("可视化已保存为 'images/transformers_library.png'")

# 4. 总结
print("\n" + "=" * 70)
print("Transformers库总结")
print("=" * 70)
print("""
Transformers库特点:
1. 丰富的预训练模型
2. 简单易用的API
3. 活跃的社区支持
""")
print("=" * 70)
print("\nTransformers Library Demo完成！")
