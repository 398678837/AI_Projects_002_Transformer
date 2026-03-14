# 价值迭代（Value Iteration）详细文档

## 1. 概念介绍

### 1.1 什么是价值迭代

价值迭代（Value Iteration）是动态规划中另一种核心算法，用于在已知环境模型的情况下找到最优策略。它直接求解贝尔曼最优方程，通过迭代更新价值函数直到收敛，然后从收敛的价值函数中提取最优策略。

### 1.2 价值迭代的核心思想

价值迭代的核心是**贝尔曼最优算子**：
$$ V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')] $$

### 1.3 与策略迭代的比较

| 特性 | 策略迭代 | 价值迭代 |
|------|----------|----------|
| 策略评估 | 需要完整评估 | 内嵌在迭代中 |
| 迭代次数 | 少 | 多 |
| 单次计算 | 大 | 小 |
| 收敛速度 | 快（迭代次数少） | 慢（迭代次数多） |

## 2. 技术原理

### 2.1 贝尔曼最优方程

价值迭代直接求解贝尔曼最优方程：
$$ V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')] $$

### 2.2 价值迭代算法

```
1. 初始化 V(s) = 0, 对所有 s
2. 重复:
   对每个状态 s:
     V_new(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) V(s')]
   如果 ||V_new - V|| < θ, 停止
   V = V_new
3. 提取策略: π(s) = argmax_a Σ P(s'|s,a) [R(s,a,s') + γ V(s')]
```

### 2.3 收敛性证明

价值迭代的更新算子是**收缩映射**：
$$ ||T(V_1) - T(V_2)||_\infty \leq \gamma ||V_1 - V_2||_\infty $$

其中T是贝尔曼最优算子，γ < 1，所以价值迭代保证收敛。

### 2.4 提取策略

价值函数收敛后，使用贪婪策略提取最优动作：
$$ \pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')] $$

## 3. 代码实现

### 3.1 核心代码

```python
def value_iteration(gamma=0.9, theta=0.001):
    V = np.zeros(n_states)
    
    while True:
        V_new = np.zeros(n_states)
        
        for state in states:
            q_values = []
            for action in actions:
                next_state = get_next_state(state, action)
                r = rewards[next_state]
                q = r + gamma * V[state_index[next_state]]
                q_values.append(q)
            V_new[state_index[state]] = max(q_values)
        
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    
    return V
```

### 3.2 策略提取

```python
def extract_policy(V, gamma):
    policy = np.zeros(n_states)
    
    for state in states:
        q_values = []
        for action in actions:
            next_state = get_next_state(state, action)
            r = rewards[next_state]
            q = r + gamma * V[state_index[next_state]]
            q_values.append(q)
        policy[state_index[state]] = np.argmax(q_values)
    
    return policy
```

## 4. 优缺点分析

### 4.1 优点

1. **简单直接**：不需要显式的策略迭代
2. **易于实现**：代码简洁
3. **保证收敛**：收缩映射保证收敛
4. **灵活性高**：可以在任何迭代次数停止

### 4.2 缺点

1. **需要模型**：必须知道转移概率
2. **迭代次数多**：通常需要更多迭代
3. **内存开销**：需要存储完整价值函数

## 5. 应用场景

- **网格世界**：路径规划问题
- **经典控制问题**：Cart-Pole、Mountain Car
- **棋类游戏**：西洋跳棋、围棋（与蒙特卡洛树搜索结合）

## 6. 总结

价值迭代是求解MDP最优策略的经典算法，通过直接迭代价值函数来找到最优策略。它实现简单，收敛保证，是强化学习的基础算法之一。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
