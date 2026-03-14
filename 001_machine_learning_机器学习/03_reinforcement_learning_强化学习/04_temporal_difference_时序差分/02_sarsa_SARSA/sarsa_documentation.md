# SARSA详细文档

## 1. 概念介绍

### 1.1 什么是SARSA

SARSA（State-Action-Reward-State-Action）是一种On-Policy的时序差分学习方法，由Rummery和Niranjan于1994年提出。它的名称来自于更新公式中包含的五个元素：State, Action, Reward, State, Action。

### 1.2 SARSA vs Q学习

| 特性 | Q学习 | SARSA |
|------|-------|-------|
| 策略类型 | Off-Policy | On-Policy |
| 更新目标 | max_a' Q(s',a') | Q(s',a') |
| 探索策略 | ε-贪婪 | ε-贪婪 |
| 收敛速度 | 较快 | 较慢 |
| 稳定性 | 较不稳定 | 较稳定 |

### 1.3 SARSA的特点

1. **On-Policy**：评估和改改进同一策略
2. **考虑探索**：更新时考虑实际采取的动作
3. **更安全**：不会学习导致高风险动作的策略
4. **适合连续任务**：在探索过程中不会崩溃

## 2. 技术原理

### 2.1 SARSA更新规则

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)] $$

其中$a_{t+1}$是实际执行的下一个动作。

### 2.2 算法流程

```
1. 初始化 Q(s,a) = 0
2. 对每个episode:
   a. 初始化状态 s, 选择动作 a (ε-贪婪)
   b. 对每一步:
      i. 执行动作 a, 观察 r, s'
      ii. 选择下一动作 a' (ε-贪婪)
      iii. 更新: Q(s,a) += α[r + γ * Q(s',a') - Q(s,a)]
      iv. s = s', a = a'
   c. 直到 s 是终止状态
```

### 2.3 收敛条件

SARSA在满足以下条件时收敛：
1. 步长参数α足够小
2. 满足GLIE条件（Greedy in the Limit with Infinite Exploration）
3. 状态-动作对被无限次访问

## 3. 代码实现

### 3.1 核心代码

```python
def sarsa(n_episodes, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(n_episodes):
        state = start_state
        
        # 初始动作选择
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        while not done:
            # 执行动作
            next_state, reward, done = env.step(state, action)
            
            # 选择下一个动作
            if np.random.random() < epsilon:
                next_action = np.random.randint(n_actions)
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA更新
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
    
    return Q
```

## 4. 优缺点分析

### 4.1 优点

1. **安全探索**：不会学习导致高惩罚的策略
2. **适合连续任务**：在持续交互中稳定学习
3. **实现简单**：与Q学习类似
4. **On-Policy学习**：理论分析相对简单

### 4.2 缺点

1. **可能不是最优**：收敛到的可能是次优策略
2. **收敛较慢**：需要更多样本
3. **对ε敏感**：ε值影响性能

## 5. 应用场景

- **机器人控制**
- **自动驾驶**
- **游戏AI**
- **资源管理**

## 6. 总结

SARSA是一种重要的On-Policy TD学习方法，特别适合需要安全探索的场景。虽然它可能不收敛到最优策略，但在许多实际应用中表现出色。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
- Rummery, G.A., & Niranjan, M. (1994). "On-Line Q-Learning Using Connectionist Systems"
