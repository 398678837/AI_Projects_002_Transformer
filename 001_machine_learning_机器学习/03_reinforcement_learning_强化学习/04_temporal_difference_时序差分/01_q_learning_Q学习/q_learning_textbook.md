# Q学习教材

## 第一章：Q学习简介

### 1.1 什么是Q学习

Q学习是强化学习中最重要的算法之一，是一种无模型的离策略学习方法。

### 1.2 核心思想

直接学习最优动作价值函数Q*(s,a)，而不需要遵循该策略。

### 1.3 特点

1. Off-Policy：评估策略与行为策略不同
2. TD学习：结合蒙特卡洛和动态规划优点
3. 异步更新：每步都能更新

## 第二章：Q学习更新规则

### 2.1 更新公式

$$ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)] $$

### 2.2 参数说明

- α：学习率
- γ：折扣因子
- max_a' Q(s',a')：下一个状态的最大Q值

## 第三章：算法流程

### 3.1 伪代码

```
1. 初始化 Q(s,a) = 0
2. 对每个episode:
   a. 初始化状态 s
   b. 对每一步:
      i. ε-贪婪选择动作 a
      ii. 执行动作, 观察 r, s'
      iii. 更新 Q(s,a)
      iv. s = s'
   c. 直到到达终止状态
```

## 第四章：探索与利用

### 4.1 ε-贪婪策略

- 概率 1-ε：选择最优动作
- 概率 ε：随机选择动作

## 第五章：代码实现

```python
def q_learning(env, n_episodes, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(n_episodes):
        state = env.reset()
        while True:
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done = env.step(action)
            
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            if done:
                break
            state = next_state
    
    return Q
```

## 第六章：总结

Q学习是强化学习的基础算法，理解它对学习其他算法很重要。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
