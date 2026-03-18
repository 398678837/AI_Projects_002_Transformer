"""
Qwen模型加载演示
Model Loading
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Qwen Model Loading Demo")
print("=" * 70)

# 1. 模型加载
print("\n1. Model Loading...")

print("""
Qwen Model Loading:
- Use AutoModelForCausalLM
- Load quantized models
- Configure generation parameters
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
models = ['Qwen-0.5B', 'Qwen-1.8B', 'Qwen-7B', 'Qwen-14B']
params = [0.5, 1.8, 7, 14]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(models, params, color=colors, alpha=0.7)
ax.set_ylabel('Parameter Count (B)', fontsize=10)
ax.set_title('Qwen Model Parameter Comparison', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, param in zip(bars, params):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{param}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
models = ['Qwen-0.5B', 'Qwen-1.8B', 'Qwen-7B', 'Qwen-14B']
gpu_memory = [1, 4, 14, 28]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(models, gpu_memory, color=colors, alpha=0.7)
ax.set_ylabel('GPU Memory (GB)', fontsize=10)
ax.set_title('Qwen Model GPU Memory Requirements', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, mem in zip(bars, gpu_memory):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{mem}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'qwen_model_loading.png'))
print("Visualization saved to 'images/qwen_model_loading.png'")

# 3. 模型加载方式
print("\n3. Model Loading Methods...")

print("""
Model Loading Methods:
1. Basic Loading:
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.8B-Chat")

2. Quantized Loading:
   - 4-bit quantization
   - 8-bit quantization
   - Lower GPU memory requirements

3. Advanced Loading:
   - Load with specific configurations
   - Use quantization tools
   - Configure generation parameters
""")

# 4. 使用方法
print("\n4. Usage Methods...")

print("""
Usage Methods:
1. Basic Usage:
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.8B-Chat")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.8B-Chat")

2. Advanced Usage:
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen-1.8B-Chat",
       torch_dtype=torch.float16,
       device_map="auto",
       load_in_4bit=True
   )

3. Generation:
   inputs = tokenizer("Hello, how are you?", return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=100)
   response = tokenizer.decode(outputs[0])
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

# 6. 总结
print("\n" + "=" * 70)
print("Model Loading Summary")
print("=" * 70)

print("""
Key Concepts:
1. Model loading is the first step in using Qwen
2. Load models from Hugging Face Hub
3. Common methods: AutoModelForCausalLM
4. Loading options: torch_dtype, device_map, load_in_4bit

Model Loading Methods:
- Basic loading
- Quantized loading
- Advanced loading

Usage Methods:
- Basic usage
- Advanced usage
- Generation

Model Loading Applications:
- Research
- Development
- Education

Loading Steps:
1. Import necessary modules
2. Load model and tokenizer
3. Configure loading options
4. Use model for generation
""")

print("=" * 70)
print("Model Loading Demo completed!")
print("=" * 70)
