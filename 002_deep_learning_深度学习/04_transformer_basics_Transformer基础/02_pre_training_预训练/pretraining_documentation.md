# 预训练详细文档

## 1. 概念介绍

### 1.1 什么是预训练

预训练是在大规模数据上训练模型，学习通用知识。

### 1.2 预训练任务

- **MLM (Masked Language Model)**: 掩码语言模型
- **CLM (Causal Language Model)**: 因果语言模型
- **Text-to-Text**: 文本到文本

## 2. 主流预训练模型

### 2.1 BERT

- 架构: Transformer Encoder
- 预训练: MLM + NSP
- 特点: 双向编码

### 2.2 GPT

- 架构: Transformer Decoder
- 预训练: CLM
- 特点: 单向生成

### 2.3 T5

- 架构: Encoder-Decoder
- 预训练: Text-to-Text
- 特点: 统一框架

## 3. 总结

预训练+微调是NLP的主流范式。
