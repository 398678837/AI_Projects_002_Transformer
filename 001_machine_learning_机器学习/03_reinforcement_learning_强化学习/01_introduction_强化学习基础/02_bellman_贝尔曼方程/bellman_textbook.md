# 贝尔曼方程教材

## 第一章：贝尔曼方程简介

### 1.1 什么是贝尔曼方程

贝尔曼方程描述了价值函数的递归关系，是强化学习的核心方程。

### 1.2 贝尔曼方程的作用

1. 形式化描述价值函数
2. 提供求解方法
3. 连接策略与价值
4. 最优性原理

## 第二章：贝尔曼期望方程

### 2.1 状态价值函数的贝尔曼期望方程

$$ V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')] $$

### 2.2 动作价值函数的贝尔曼期望方程

$$ Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')] $$

## 第三章：贝尔曼最优方程

### 3.1 状态价值函数的贝尔曼最优方程

$$ V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')] $$

### 3.2 动作价值函数的贝尔曼最优方程

$$ Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')] $$

## 第四章：求解方法

### 4.1 策略评估

求解贝尔曼期望方程，迭代更新直到收敛。

### 4.2 值迭代

求解贝尔曼最优方程，每次迭代取最大值。

### 4.3 策略迭代

1. 策略评估
2. 策略改进
3. 重复

## 第五章：实践

```python
# 值迭代
def value_iteration(V, gamma):
    V_new = np.zeros_like(V)
    for state in states:
        q_values = []
        for action in actions:
            next_state = get_next_state(state, action)
            r = rewards[next_state]
            q = r + gamma * V[next_state]
            q_values.append(q)
        V_new[state] = max(q_values)
    return V_new
```

## 第六章：总结

贝尔曼方程是强化学习的数学基础，理解它对于学习强化学习至关重要。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
