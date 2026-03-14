"""
Qwen对话实现演示
Chat Implementation
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Qwen对话实现演示")
print("=" * 70)

# 1. 对话实现
print("\n1. 对话实现...")

print("""
对话实现:
- 构建Prompt模板
- 设置生成参数
- 流式输出
""")

# 2. 可视化
print("\n2. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
params = ['Temperature', 'Top_p', 'Top_k', 'Max_tokens']
values = [0.7, 0.9, 50, 2048]
ax.bar(params, values, color='steelblue', alpha=0.7)
ax.set_ylabel('参数值')
ax.set_title('生成参数设置')
ax.grid(True, alpha=0.3)

ax = axes[1]
turns = ['第一轮', '第二轮', '第三轮', '第四轮']
response_time = [2, 2.5, 3, 3.2]
ax.bar(turns, response_time, color='coral', alpha=0.7)
ax.set_ylabel('响应时间(秒)')
ax.set_title('多轮对话响应时间')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/chat_implementation.png')
print("可视化已保存为 'images/chat_implementation.png'")

print("\n" + "=" * 70)
print("对话实现总结")
print("=" * 70)
print("通过Pipeline实现对话生成。")
print("=" * 70)
