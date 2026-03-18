"""
模型加载演示
Model Loading
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Model Loading Demo")
print("=" * 70)

# 1. 模型加载概念
print("\n1. Model Loading Concept...")

print("""
Model Loading:
- Load models from Hugging Face Hub
- from_pretrained() / AutoModel / Pipeline
- Support multiple formats
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = ['from_pretrained()', 'AutoModel', 'Pipeline']
loading_times = [15, 12, 8]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(methods, loading_times, color=colors, alpha=0.7)
ax.set_ylabel('Loading Time (s)', fontsize=10)
ax.set_title('Loading Time by Method', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, time in zip(bars, loading_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{time}s', ha='center', va='bottom', fontsize=9)

ax = axes[1]
categories = ['BERT', 'GPT-2', 'T5', 'Llama', 'Qwen']
model_sizes = [400, 500, 300, 7000, 1800]
colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
bars = ax.bar(categories, model_sizes, color=colors, alpha=0.7)
ax.set_ylabel('Model Size (MB)', fontsize=10)
ax.set_title('Model Size by Model', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, size in zip(bars, model_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
            f'{size:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'model_loading.png'))
print("Visualization saved to 'images/model_loading.png'")

# 3. 模型加载方式
print("\n3. Model Loading Methods...")

print("""
Model Loading Methods:
1. from_pretrained()
   - Direct model loading
   - Auto download weights
   - Support multiple formats

2. AutoModel
   - Auto select model class
   - Simple API
   - Compatible with multiple frameworks

3. Pipeline
   - High-level interface
   - One-click inference
   - Support multiple tasks
""")

# 4. 使用方法
print("\n4. Usage Methods...")

print("""
Usage Methods:
1. Basic Usage:
   from transformers import AutoModel, AutoTokenizer
   model = AutoModel.from_pretrained("bert-base-uncased")
   tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

2. Advanced Usage:
   from transformers import AutoModelForSequenceClassification
   model = AutoModelForSequenceClassification.from_pretrained(
       "distilbert-base-uncased-finetuned-sst-2-english",
       num_labels=2
   )

3. Loading Options:
   model = AutoModel.from_pretrained(
       "bert-base-uncased",
       torchscript=True,
       cache_dir="./cache",
       force_download=False
   )

4. Loading Optimization:
   model = AutoModel.from_pretrained(
       "bert-base-uncased",
       torch_dtype=torch.float16,
       device_map="auto",
       load_in_8bit=True
   )
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Model Loading Applications:
1. Research:
   - Quick experiments
   - Model comparison
   - Baseline models

2. Development:
   - Quick prototyping
   - Production deployment
   - Model integration

3. Education:
   - Teaching examples
   - Project reference
   - Learning resources
""")

# 6. 注意事项
print("\n6. Precautions...")

print("""
Precautions:
1. Network Connection:
   - Ensure network connection
   - Use proxy (if needed)
   - Check firewall settings

2. Disk Space:
   - Ensure sufficient disk space
   - Clean cache
   - Use small models

3. Memory Requirements:
   - Ensure sufficient memory
   - Use small models
   - Use quantization

4. Version Compatibility:
   - Check version compatibility
   - Update libraries
   - Check dependencies
""")

# 7. 总结
print("\n" + "=" * 70)
print("Model Loading Summary")
print("=" * 70)

print("""
Key Concepts:
1. Model loading is the first step in using pre-trained models
2. Load models from Hugging Face Hub
3. Common methods: from_pretrained(), AutoModel, Pipeline
4. Loading options: torchscript, cache_dir, force_download

Model Loading Methods:
- from_pretrained()
- AutoModel
- Pipeline

Usage Methods:
- Basic usage
- Advanced usage
- Loading options
- Loading optimization

Model Loading Applications:
- Research
- Development
- Education

Precautions:
- Network connection
- Disk space
- Memory requirements
- Version compatibility
""")

print("=" * 70)
print("Model Loading Demo completed!")
print("=" * 70)
