"""
Qwen部署演示
Deployment
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Qwen部署演示")
print("=" * 70)

# 1. 部署方式
print("\n1. 部署方式...")

print("""
部署方式:
- 本地部署
- 云端部署
- API服务
- Docker容器
""")

# 2. 可视化
print("\n2. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
methods = ['本地部署', '云端API', 'Docker', 'Serverless']
latency = [50, 200, 80, 150]
ax.bar(methods, latency, color='steelblue', alpha=0.7)
ax.set_ylabel('延迟(ms)')
ax.set_title('不同部署方式延迟')
ax.grid(True, alpha=0.3)

ax = axes[1]
cost = [0, 100, 30, 50]
ax.bar(methods, cost, color='coral', alpha=0.7)
ax.set_ylabel('成本(相对)')
ax.set_title('不同部署方式成本')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/deployment.png')
print("可视化已保存为 'images/deployment.png'")

print("\n" + "=" * 70)
print("部署总结")
print("=" * 70)
print("根据需求选择合适的部署方式。")
print("=" * 70)
