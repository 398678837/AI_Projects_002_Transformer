# Actor-Critic教材

## 第一章：Actor-Critic简介

### 1.1 什么是Actor-Critic

结合价值函数与策略梯度的算法。

### 1.2 组成

- Actor：策略网络
- Critic：价值网络

## 第二章：技术原理

### 2.1 优势函数

$$ A(s,a) = Q(s,a) - V(s) $$

### 2.2 TD误差

$$ \delta = r + \gamma V(s') - V(s) $$

## 第三章：算法流程

```
1. 采样
2. 计算TD误差
3. 更新Critic
4. 更新Actor
```

## 第四章：总结

Actor-Critic是现代强化学习的基础。
