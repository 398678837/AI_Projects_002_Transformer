# 策略迭代教材

## 第一章：策略迭代简介

### 1.1 什么是策略迭代

策略迭代是动态规划中的一种算法，用于在已知环境模型的情况下找到最优策略。

### 1.2 核心思想

策略迭代通过两个步骤交替进行：
1. **策略评估**：计算当前策略的价值函数
2. **策略改进**：基于价值函数更新策略

## 第二章：策略评估

### 2.1 什么是策略评估

策略评估是计算给定策略π的状态价值函数V^π。

### 2.2 迭代公式

$$ V_{k+1}^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k^{\pi}(s')] $$

## 第三章：策略改进

### 3.1 什么是策略改进

策略改进是基于当前价值函数，选择更好动作的过程。

### 3.2 改进方法

使用贪婪策略：
$$ \pi'(s) = \arg\max_a Q^{\pi}(s, a) $$

## 第四章：策略迭代算法

### 4.1 算法流程

```
1. 初始化随机策略 π
2. 循环直到策略收敛:
   a. 策略评估: 计算 V^π
   b. 策略改进: 更新 π
3. 返回最优策略
```

### 4.2 收敛性

- 策略单调不降
- 有限步收敛到最优策略

## 第五章：代码实现

```python
# 策略评估
def policy_evaluation(policy):
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_new = np.zeros(n_states)
        for state in states:
            v = 0
            for a_idx, action in enumerate(actions):
                next_state = get_next_state(state, action)
                r = rewards[next_state]
                v += policy[a_idx] * (r + gamma * V[state_index[next_state]])
            V_new[state_index[state]] = v
        V = V_new
    return V

# 策略改进
def policy_improvement(V):
    policy = np.zeros(n_actions)
    for state in states:
        q_values = [Q(s, a) for a in actions]
        best = np.argmax(q_values)
        policy[best] = 1.0
    return policy
```

## 第六章：总结

策略迭代通过交替进行策略评估和策略改进来找到最优策略。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
