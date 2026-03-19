# Prompt工程+CoT思维链 Demo项目

## 项目说明

**项目名称**：Prompt-Engineering-CoT-Demo  
**目标**：2天掌握Prompt工程与CoT思维链核心技巧  
**适用人群**：AI工程师、开发者、研究人员  

---

## 📅 2天学习计划

### Day 1：学习核心概念与模板
- **10分钟**：理解Prompt工程、Few-Shot、CoT核心概念
- **30分钟**：背熟3个万能模板
- **1小时**：理解模板应用场景

### Day 2：项目开发与实践
- **2小时**：创建多场景Prompt模板库
- **2小时**：实现CoT数学/逻辑推理Demo
- **30分钟**：上传GitHub、更新简历

---

## 📁 项目结构

```
101_prompt_engineering_CoT_Prompt工程/
├── README.md              # 项目说明（本文件）
├── prompt_templates.md    # 多场景Prompt模板库
├── cot_demo.py            # CoT推理脚本
└── requirements.txt        # 依赖文件
```

---

## 🎯 核心概念

### 1. Prompt工程
给AI清晰、结构化指令，让输出更准确、符合预期

### 2. Few-Shot（少样本学习）
给AI 2-3个例子，让AI照着格式输出

### 3. CoT（思维链）
让AI**先写思考过程，再给答案**，大幅提升推理准确率

---

## 📝 必背万能模板

### ① 标准指令模板
```
角色：{你要AI扮演的角色}
任务：{具体要做什么}
要求：
1. {格式要求}
2. {细节/禁忌}
输出：{期望格式}
```

### ② Few-Shot 模板
```
任务：{任务}
示例1：
输入：{...}
输出：{...}
示例2：
输入：{...}
输出：{...}
请按示例格式输出：
```

### ③ CoT 思维链模板（最核心）
```
问题：{题目}
要求：
1. 先一步步写出思考过程
2. 最后给出最终答案
思考过程：
最终答案：
```

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行CoT Demo
```bash
python cot_demo.py
```

### 3. 查看模板库
打开 `prompt_templates.md` 查看所有模板

---

## 📊 学习成果

✅ 掌握Prompt工程核心技巧  
✅ 搭建多场景Prompt模板库  
✅ 实现基于思维链的数学与逻辑推理  
✅ 可显著提升大模型输出准确性与推理能力  

---

## 💼 简历写法

**掌握Prompt工程、CoT思维链核心技巧，搭建多场景Prompt模板库，实现基于思维链的数学与逻辑推理，可显著提升大模型输出准确性与推理能力。**

---

## 🎓 学习路径

1. **基础阶段**（1天）：学习Prompt工程、Few-Shot、CoT
2. **实践阶段**（1天）：创建模板库、实现CoT推理
3. **应用阶段**：将技巧应用到实际项目中

---

## 📚 参考资料

- OpenAI Prompt Engineering Guide
- Google CoT Paper
- Hugging Face Transformers Documentation

---

## 🤝 贡献

欢迎提交Issue和PR！

---

## 📄 License

MIT License