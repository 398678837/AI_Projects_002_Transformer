# Transformer vs RNN详细文档

## 1. 概念介绍

### 1.1 注意力机制

注意力机制模拟人脑的注意力，动态分配计算资源。

### 1.2 核心概念

- **Query**: 我要找什么
- **Key**: 哪些位置有相关信息
- **Value**: 相关信息的内容

## 2. Transformer vs RNN

### 2.1 并行计算

- RNN: 顺序计算，无法并行
- Transformer: 支持并行计算

### 2.2 长距离依赖

- RNN: 梯度消失，难以捕获
- Transformer: 注意力机制直接建模

### 2.3 计算复杂度

- RNN: O(n)
- Transformer: O(n²)

## 3. 总结

Transformer在多数场景优于RNN，已成为NLP主流。
