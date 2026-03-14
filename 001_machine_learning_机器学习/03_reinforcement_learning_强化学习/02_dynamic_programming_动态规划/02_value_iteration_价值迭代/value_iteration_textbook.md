# 价值迭代教材

## 第一章：价值迭代简介

### 1.1 什么是价值迭代

价值迭代是动态规划中的一种算法，直接求解贝尔曼最优方程来找到最优策略。

### 1.2 核心思想

通过迭代更新价值函数，直到收敛，然后提取最优策略。

## 第二章：贝尔曼最优方程

### 2.1 最优状态价值函数

$$ V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')] $$

### 2.2 最优动作价值函数

$$ Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')] $$

## 第三章：价值迭代算法

### 3.1 算法流程

```
1. 初始化 V(s) = 0
2. 重复直到收敛:
   对每个状态 s:
     V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) V(s')]
3. 提取策略: π(s) = argmax_a Q(s,a)
```

### 3.2 收敛性

- 贝尔曼最优算子是收缩映射
- 一定收敛到最优价值函数

## 第四章：与策略迭代比较

| 特性 | 策略迭代 | 价值迭代 |
|------|----------|----------|
| 策略评估 | 需要完整评估 | 内嵌在迭代中 |
| 迭代次数 | 少 | 多 |
| 单次计算 | 大 | 小 |

## 第五章：代码实现

```python
def value_iteration():
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

## 第六章：总结

价值迭代直接求解贝尔曼最优方程，是强化学习的基础算法。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
