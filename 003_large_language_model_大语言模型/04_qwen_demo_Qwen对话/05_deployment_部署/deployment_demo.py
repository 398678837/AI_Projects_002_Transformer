"""
Qwen部署演示
Deployment
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Qwen Deployment Demo")
print("=" * 70)

# 1. 部署方式
print("\n1. Deployment Methods...")

print("""
Deployment Methods:
- Local deployment
- Cloud deployment
- API service
- Docker container
""")

# 2. 可视化
print("\n2. Visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
methods = ['Local', 'Cloud API', 'Docker', 'Serverless']
latency = [50, 200, 80, 150]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(methods, latency, color=colors, alpha=0.7)
ax.set_ylabel('Latency (ms)', fontsize=10)
ax.set_title('Latency by Deployment Method', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, lat in zip(bars, latency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{lat}', ha='center', va='bottom', fontsize=9)

ax = axes[1]
cost = [0, 100, 30, 50]
colors = ['steelblue', 'coral', 'green', 'orange']
bars = ax.bar(methods, cost, color=colors, alpha=0.7)
ax.set_ylabel('Cost (Relative)', fontsize=10)
ax.set_title('Cost by Deployment Method', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, cst in zip(bars, cost):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{cst}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'deployment.png'))
print("Visualization saved to 'images/deployment.png'")

# 3. 部署步骤
print("\n3. Deployment Steps...")

print("""
Deployment Steps:
1. Local Deployment:
   - Install dependencies
   - Run model locally

2. Cloud Deployment:
   - Use Hugging Face Inference API
   - Configure API key
   - Call API

3. Docker Deployment:
   - Create Dockerfile
   - Build Docker image
   - Run container

4. Serverless Deployment:
   - Use AWS Lambda
   - Configure triggers
   - Deploy function
""")

# 4. 注意事项
print("\n4. Precautions...")

print("""
Precautions:
1. Performance Optimization:
   - Use quantization
   - Use caching
   - Use optimization tools

2. Security:
   - API key management
   - Data encryption
   - Access control

3. Cost Control:
   - Choose appropriate model
   - Use auto-scaling
   - Monitor usage
""")

# 5. 应用场景
print("\n5. Applications...")

print("""
Deployment Applications:
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
print("Deployment Summary")
print("=" * 70)

print("""
Key Concepts:
1. Deployment is the final step in LLM applications
2. Choose appropriate deployment method
3. Optimize performance
4. Ensure security

Deployment Methods:
- Local deployment
- Cloud deployment
- Docker deployment
- Serverless deployment

Deployment Steps:
- Local deployment
- Cloud deployment
- Docker deployment
- Serverless deployment

Deployment Applications:
- Research
- Development
- Education

Precautions:
- Performance optimization
- Security
- Cost control
""")

print("=" * 70)
print("Deployment Demo completed!")
print("=" * 70)
