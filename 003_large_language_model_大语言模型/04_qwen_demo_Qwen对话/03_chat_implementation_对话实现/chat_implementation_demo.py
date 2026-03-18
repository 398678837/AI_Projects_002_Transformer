"""
Qwen对话实现演示
Chat Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Qwen Chat Implementation Demo")
print("=" * 70)

# 1. 对话实现
print("\n1. Chat Implementation...")

print("""
Chat Implementation:
- Build Prompt templates
- Set generation parameters
- Stream output
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
params = ['Temperature', 'Top_p', 'Top_k', 'Max_tokens']
values = [0.7, 0.9, 50, 2048]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(params, values, color=colors, alpha=0.7)
ax.set_ylabel('Parameter Value', fontsize=10)
ax.set_title('Generation Parameter Settings', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{value}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
turns = ['Turn 1', 'Turn 2', 'Turn 3', 'Turn 4']
response_time = [2, 2.5, 3, 3.2]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(turns, response_time, color=colors, alpha=0.7)
ax.set_ylabel('Response Time (s)', fontsize=10)
ax.set_title('Multi-turn Conversation Response Time', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, time in zip(bars, response_time):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{time:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'chat_implementation.png'))
print("Visualization saved to 'images/chat_implementation.png'")

# 3. 对话实现步骤
print("\n3. Chat Implementation Steps...")

print("""
Chat Implementation Steps:
1. Build Prompt:
   prompt = "User: Hello\nAssistant:"

2. Set Generation Parameters:
   - Temperature: Control randomness
   - Top_p: Nucleus sampling
   - Top_k: Top-k sampling

3. Stream Output:
   - Real-time output
   - Token-by-token generation
""")

# 4. 使用方法
print("\n4. Usage Methods...")

print("""
Usage Methods:
1. Basic Usage:
   from transformers import pipeline
   chat = pipeline("conversational", model="Qwen/Qwen-1.8B-Chat")

2. Advanced Usage:
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.8B-Chat")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.8B-Chat")

3. Generation:
   inputs = tokenizer(prompt, return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=100)
   response = tokenizer.decode(outputs[0])
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Chat Implementation Applications:
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
print("Chat Implementation Summary")
print("=" * 70)

print("""
Key Concepts:
1. Chat implementation is the core of LLM applications
2. Build Prompt templates
3. Set generation parameters
4. Stream output

Chat Implementation Steps:
- Build Prompt
- Set parameters
- Stream output

Usage Methods:
- Basic usage
- Advanced usage
- Generation

Chat Implementation Applications:
- Research
- Development
- Education

Implementation Steps:
1. Import necessary modules
2. Load model and tokenizer
3. Build Prompt
4. Generate response
""")

print("=" * 70)
print("Chat Implementation Demo completed!")
print("=" * 70)
