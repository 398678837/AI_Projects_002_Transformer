# 多头注意力详细文档

## 1. 概念介绍

### 1.1 什么是多头注意力

多头注意力(Multi-Head Attention)使用多个注意力头并行计算，每个头关注不同的信息。

### 1.2 核心思想

- 多个Q、K、V矩阵
- 并行计算多个注意力
- 拼接所有头的输出

## 2. 计算过程

### 2.1 公式

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

## 3. 优点

- 捕获多种语义关系
- 增强表达能力
- 并行计算

## 4. 总结

多头注意力是Transformer的核心组件。
