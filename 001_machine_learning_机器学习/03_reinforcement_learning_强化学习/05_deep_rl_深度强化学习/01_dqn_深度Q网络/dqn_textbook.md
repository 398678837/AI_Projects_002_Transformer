# 深度Q网络（DQN）教材

## 第一章：DQN简介

### 1.1 什么是DQN

DQN是深度学习与Q学习结合的产物，使用神经网络近似Q函数。

### 1.2 核心创新

1. 经验回放
2. 目标网络

## 第二章：技术原理

### 2.1 Q网络

$$ Q(s,a;\theta) \approx Q^*(s,a) $$

### 2.2 经验回放

存储交互数据，随机采样学习。

### 2.3 目标网络

$$ Y_t = r_t + \gamma \max_{a'} Q(s_{t+1},a';\theta^-) $$

### 2.4 损失函数

$$ L(\theta) = \mathbb{E}[(Y_t - Q(s_t,a_t;\theta))^2] $$

## 第三章：算法流程

```
1. 初始化Q网络和目标网络
2. 对每个episode:
   a. 选择动作 (ε-贪婪)
   b. 执行动作，存储经验
   c. 随机采样更新网络
   d. 定期更新目标网络
```

## 第四章：改进算法

- Double DQN
- Dueling DQN
- Prioritized Experience Replay

## 第五章：总结

DQN是深度强化学习的基础算法。

---

**参考资料**：
- Mnih et al., 2015, Nature
