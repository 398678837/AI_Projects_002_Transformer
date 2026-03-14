# Markov决策过程教材

## 第一章：MDP简介

### 1.1 什么是MDP

MDP（Markov Decision Process）是强化学习的数学框架，用于描述智能体在环境中的决策过程。

### 1.2 MDP五要素

1. **状态空间（S）**：所有可能的状态
2. **动作空间（A）**：所有可能的动作
3. **状态转移函数（P）**：$P(s'|s,a)$
4. **奖励函数（R）**：$R(s,a,s')$
5. **策略（π）**：$\pi(a|s)$

### 1.3 Markov性质

下一个状态只依赖于当前状态和动作，不依赖于历史。

## 第二章：价值函数

### 2.1 状态价值函数

$V^\pi(s)$：在状态$s$下遵循策略$\pi$的预期回报。

### 2.2 动作价值函数

$Q^\pi(s,a)$：在状态$s$下执行动作$a$，然后遵循策略$\pi$的预期回报。

### 2.3 回报（Return）

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... $$

$\gamma$是折扣因子，$\gamma \in [0,1]$。

## 第三章：Bellman方程

### 3.1 Bellman期望方程

状态价值函数：
$$ V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')] $$

## 第四章：实践

```python
# 定义状态
states = [(i, j) for i in range(4) for j in range(4)]

# 定义动作
actions = ['up', 'down', 'left', 'right']

# 随机策略
def random_policy(state):
    return np.random.choice(actions)
```

## 第五章：总结

MDP是强化学习的数学基础，理解MDP对于学习强化学习至关重要。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
