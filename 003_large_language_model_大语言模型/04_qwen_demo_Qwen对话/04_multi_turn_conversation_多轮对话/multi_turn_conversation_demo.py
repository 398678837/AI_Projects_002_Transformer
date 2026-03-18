"""
Qwen多轮对话演示
Multi-turn Conversation
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Qwen Multi-turn Conversation Demo")
print("=" * 70)

# 1. 多轮对话
print("\n1. Multi-turn Conversation...")

print("""
Multi-turn Conversation:
- Maintain conversation history
- Context understanding
- State management
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
context_length = [512, 1024, 2048, 4096]
memory_usage = [2, 4, 8, 16]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(context_length, memory_usage, color=colors, alpha=0.7)
ax.set_xlabel('Context Length (tokens)', fontsize=10)
ax.set_ylabel('Memory Usage (GB)', fontsize=10)
ax.set_title('Memory Usage by Context Length', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, mem in zip(bars, memory_usage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{mem}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
turns = ['Turn 1', 'Turn 2', 'Turn 3', 'Turn 4']
response_time = [1.5, 1.8, 2.2, 2.5]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(turns, response_time, color=colors, alpha=0.7)
ax.set_xlabel('Conversation Turn', fontsize=10)
ax.set_ylabel('Response Time (s)', fontsize=10)
ax.set_title('Response Time by Turn', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, time in zip(bars, response_time):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
            f'{time:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'multi_turn_conversation.png'))
print("Visualization saved to 'images/multi_turn_conversation.png'")

# 3. 多轮对话原理
print("\n3. Multi-turn Conversation Principles...")

print("""
Multi-turn Conversation Principles:
1. Conversation History:
   - Maintain conversation history
   - Context understanding
   - State management

2. Context Window:
   - 4096 tokens
   - 8192 tokens
   - Longer context

3. State Management:
   - Maintain conversation state
   - Context understanding
   - Topic tracking
""")

# 4. 使用方法
print("\n4. Usage Methods...")

print("""
Usage Methods:
1. Basic Usage:
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.8B-Chat")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.8B-Chat")

2. Conversation History:
   conversation = [
       {"role": "user", "content": "Hello"},
       {"role": "assistant", "content": "Hello! How can I help you?"},
       {"role": "user", "content": "I want to learn machine learning"}
   ]

3. Generate Response:
   inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=100)
   response = tokenizer.decode(outputs[0])
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Multi-turn Conversation Applications:
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
1. Memory Limitations:
   - Control conversation length
   - Use truncation
   - Optimize memory

2. Context Truncation:
   - Keep important information
   - Remove redundancy
   - Optimize context

3. Topic Drift:
   - Topic tracking
   - Context understanding
   - State management
""")

# 7. 总结
print("\n" + "=" * 70)
print("Multi-turn Conversation Summary")
print("=" * 70)

print("""
Key Concepts:
1. Multi-turn conversation is an important form of LLM applications
2. Maintain conversation history
3. Context understanding
4. State management

Multi-turn Conversation Principles:
- Conversation history
- Context window
- State management

Usage Methods:
- Basic usage
- Conversation history
- Generate response

Multi-turn Conversation Applications:
- Research
- Development
- Education

Precautions:
- Memory limitations
- Context truncation
- Topic drift
""")

print("=" * 70)
print("Multi-turn Conversation Demo completed!")
print("=" * 70)
