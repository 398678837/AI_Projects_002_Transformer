# 蒙特卡洛方法教材

## 第一章：蒙特卡洛方法简介

### 1.1 什么是蒙特卡洛方法

蒙特卡洛方法是强化学习中的无模型学习方法，通过生成完整的episode来估计价值函数。

### 1.2 核心思想

用样本均值估计期望值：
$$ V(s) \approx \frac{1}{N} \sum_{i=1}^{N} G_i(s) $$

### 1.3 特点

1. 无模型：不需要转移概率
2. 基于样本：通过交互生成样本
3. 完整episode：需要完整的轨迹

## 第二章：回报

### 2.1 回报定义

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... $$

### 2.2 首次访问vs每次访问

- 首次访问：只对首次出现更新
- 每次访问：每次出现都更新

## 第三章：蒙特卡洛策略评估

### 3.1 算法流程

```
1. 初始化 V(s) = 0
2. 对每个episode:
   a. 生成完整episode
   b. 对每个状态首次出现:
      计算回报 G
      V(s) += (G - V(s)) / N(s)
```

### 3.2 增量式更新

$$ V(s) \leftarrow V(s) + \alpha [G - V(s)] $$

## 第四章：代码实现

```python
def mc_policy_evaluation(policy, n_episodes):
    V = {}
    for episode in generate_episodes(policy, n_episodes):
        states_visited = set()
        for i, (state, reward) in enumerate(episode):
            if state not in states_visited:
                G = sum(r for _, r in episode[i:])
                V[state] = V.get(state, 0) + alpha * (G - V.get(state, 0))
    return V
```

## 第五章：总结

蒙特卡洛方法是强化学习的基础无模型方法。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
