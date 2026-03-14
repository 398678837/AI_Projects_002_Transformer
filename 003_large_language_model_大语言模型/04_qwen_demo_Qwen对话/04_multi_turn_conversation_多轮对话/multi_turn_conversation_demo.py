"""
Qwen多轮对话演示
Multi-turn Conversation
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Qwen多轮对话演示")
print("=" * 70)

# 1. 多轮对话
print("\n1. 多轮对话...")

print("""
多轮对话:
- 维护对话历史
- 上下文理解
- 状态管理
""")

# 2. 可视化
print("\n2. 可视化...")

fig, ax = plt.subplots(figsize=(10, 5))

context_length = [512, 1024, 2048, 4096]
memory_usage = [2, 4, 8, 16]
ax.bar(range(len(context_length)), memory_usage, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(context_length)))
ax.set_xticklabels([f'{c} tokens' for c in context_length])
ax.set_ylabel('内存占用(GB)')
ax.set_title('不同上下文长度的内存需求')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/multi_turn_conversation.png')
print("可视化已保存为 'images/multi_turn_conversation.png'")

print("\n" + "=" * 70)
print("多轮对话总结")
print("=" * 70)
print("多轮对话需要维护对话历史。")
print("=" * 70)
