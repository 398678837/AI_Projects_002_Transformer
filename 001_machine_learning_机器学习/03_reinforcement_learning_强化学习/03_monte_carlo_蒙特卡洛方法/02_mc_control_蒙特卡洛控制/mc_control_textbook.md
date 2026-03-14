# 蒙特卡洛控制教材

## 第一章：蒙特卡洛控制简介

### 1.1 什么是蒙特卡洛控制

蒙特卡洛控制是利用蒙特卡洛方法学习最优策略的过程。

### 1.2 核心思想

平衡探索与利用：
- 探索：尝试新动作
- 利用：使用已知最优策略

## 第二章：ε-贪婪策略

### 2.1 定义

$$ \pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|A|} & a = \arg\max Q(s,a') \\ \frac{\epsilon}{|A|} & \text{否则} \end{cases} $$

### 2.2 探索与利用

- 高ε：更多探索
- 低ε：更多利用

## 第三章：蒙特卡洛控制算法

### 3.1 算法流程

```
1. 初始化 Q(s,a) = 0
2. 重复:
   a. 生成episode (使用ε-贪婪策略)
   b. 对每个状态-动作对:
      计算回报 G
      更新 Q(s,a)
   c. 基于Q改进策略
```

### 3.2 GLIE条件

- 无限次访问每个状态-动作对
- 策略收敛到贪婪策略

## 第四章：代码实现

```python
def mc_control(n_episodes, epsilon=0.1):
    Q = {}
    for episode in generate_episodes(n_episodes):
        G = 0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (G - Q.get((state, action), 0))
    return Q
```

## 第五章：总结

蒙特卡洛控制是重要的无模型控制方法。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
