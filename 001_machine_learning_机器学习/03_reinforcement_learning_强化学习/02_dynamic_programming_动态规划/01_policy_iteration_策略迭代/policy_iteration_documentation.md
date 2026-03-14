# 策略迭代（Policy Iteration）详细文档

## 1. 概念介绍

### 1.1 什么是策略迭代

策略迭代（Policy Iteration）是动态规划中的一种核心算法，用于在已知环境模型的情况下找到最优策略。它通过交替进行**策略评估**和**策略改进**来逐步优化策略，最终收敛到最优策略。

### 1.2 策略迭代的核心思想

策略迭代基于一个重要的理论：**给定一个策略，我们可以评估它的价值函数；给定一个价值函数，我们可以改进策略**。

### 1.3 策略迭代的适用场景

- **基于模型的强化学习**：当环境转移概率P和奖励函数R已知时
- **小规模状态空间**：状态数量不太多，可以进行完整的价值函数评估
- **需要精确解**：动态规划方法可以得到精确的最优解

## 2. 技术原理

### 2.1 策略迭代的两个步骤

#### 步骤1：策略评估（Policy Evaluation）

给定策略π，计算状态价值函数V^π：

$$ V^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^{\pi}(s')] $$

**迭代方法**：
$$ V_{k+1}^{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k^{\pi}(s')] $$

直到 $||V_{k+1}^{\pi} - V_k^{\pi}|| < \theta$

#### 步骤2：策略改进（Policy Improvement）

基于当前的价值函数V^π，改进策略：

$$ \pi'(a|s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^{\pi}(s')] $$

或使用贪婪策略：
$$ \pi'(s) = \arg\max_a Q^{\pi}(s, a) $$

### 2.2 策略迭代算法流程

```
1. 初始化策略 π (可以是随机策略)
2. 循环直到策略收敛:
   a. 策略评估: 计算 V^π
   b. 策略改进: 更新 π
   c. 如果策略不再改变, 停止
3. 返回最优策略 π* 和价值函数 V*
```

### 2.3 策略改进定理

**策略改进定理**：如果新策略π'的价值函数满足：
$$ Q^{\pi}(s, \pi'(s)) \geq V^{\pi}(s), \forall s $$

那么新策略π'至少和原策略π一样好。

### 2.4 策略迭代的收敛性

- **策略单调性**：每一步改进后，策略不会变差
- **有限步收敛**：对于有限的MDP，策略迭代会在有限步内收敛到最优策略

## 3. 代码实现

### 3.1 核心函数

```python
# 策略评估
def policy_evaluation(policy, max_iter=1000):
    V = np.zeros(n_states)
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        for state in states:
            v = 0
            for a_idx, action in enumerate(actions):
                next_state = get_next_state(state, action)
                r = rewards[next_state]
                v += policy[a_idx] * (r + gamma * V[state_index[next_state]])
            V_new[state_index[state]] = v
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new
    return V

# 策略改进
def policy_improvement(V):
    policy = np.zeros(n_actions)
    for state in states:
        q_values = []
        for action in actions:
            next_state = get_next_state(state, action)
            r = rewards[next_state]
            q = r + gamma * V[state_index[next_state]]
            q_values.append(q)
        best_action = np.argmax(q_values)
        policy[best_action] = 1.0
    return policy
```

### 3.2 完整算法

```python
def policy_iteration():
    policy = np.ones(n_actions) / n_actions  # 初始化随机策略
    
    while True:
        V = policy_evaluation(policy)        # 策略评估
        new_policy = policy_improvement(V)   # 策略改进
        
        if np.allclose(policy, new_policy):   # 检查收敛
            break
        policy = new_policy
    
    return V, policy
```

## 4. 优缺点分析

### 4.1 优点

1. **理论保证**：保证收敛到最优策略
2. **易于理解**：概念清晰，实现简单
3. **效率较高**：对于小规模问题收敛较快
4. **可解释性**：可以观察策略迭代过程

### 4.2 缺点

1. **需要完整模型**：必须知道转移概率和奖励函数
2. **计算量大**：每次迭代需要进行完整的策略评估
3. **不适用于大规模问题**：状态空间大时计算开销大

## 5. 总结

策略迭代是强化学习中基于模型的经典算法，通过交替进行策略评估和策略改进来找到最优策略。虽然它需要完整的环境模型，但对于小规模问题非常有效。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
