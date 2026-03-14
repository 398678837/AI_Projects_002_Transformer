"""
LLM应用演示
大语言模型应用场景
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LLM应用演示")
print("=" * 70)

# 1. LLM应用
print("\n1. LLM应用场景...")

applications = {
    '文本生成': ['文章写作', '代码生成', '诗歌创作', '邮件撰写'],
    '问答系统': ['知识问答', '客服机器人', '搜索引擎', '学习助手'],
    '翻译': ['语言翻译', '代码翻译', '专业术语'],
    '分析': ['情感分析', '文本摘要', '数据提取', '市场分析'],
    '编程': ['代码补全', 'Bug修复', '代码审查', '技术文档']
}

# 2. 可视化
print("\n2. 应用场景可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
apps = list(applications.keys())
usage = [85, 90, 60, 75, 80]
growth = [95, 92, 70, 78, 88]

x = np.arange(len(apps))
width = 0.35

ax.bar(x - width/2, usage, width, label='当前使用率', color='steelblue', alpha=0.7)
ax.bar(x + width/2, growth, width, label='增长潜力', color='coral', alpha=0.7)
ax.set_ylabel('百分比')
ax.set_title('LLM应用场景')
ax.set_xticks(x)
ax.set_xticklabels(apps, rotation=15)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
models = ['GPT-4', 'Claude', 'Gemini', 'Llama', 'Qwen']
capabilities = [95, 93, 91, 85, 80]
cost = [20, 15, 12, 5, 3]

ax.scatter(cost, capabilities, s=[200, 180, 160, 140, 120], 
          c=['steelblue', 'coral', 'green', 'purple', 'orange'], alpha=0.7)

for i, model in enumerate(models):
    ax.annotate(model, (cost[i], capabilities[i]), 
               textcoords="offset points", xytext=(5, 5), fontsize=10)

ax.set_xlabel('成本 (相对)')
ax.set_ylabel('能力评分')
ax.set_title('LLM模型对比')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/llm_applications.png')
print("可视化已保存为 'images/llm_applications.png'")

# 3. 总结
print("\n" + "=" * 70)
print("LLM应用总结")
print("=" * 70)
print("""
主流应用:

1. 文本生成:
   - ChatGPT用于对话
   - 代码助手Copilot

2. 问答系统:
   - 搜索引擎集成
   - 客服自动化

3. 编程辅助:
   - 代码补全
   - Bug修复

4. 分析任务:
   - 情感分析
   - 文档摘要

LLM正在改变工作方式。
""")
print("=" * 70)
print("\nLLM Applications Demo完成！")
