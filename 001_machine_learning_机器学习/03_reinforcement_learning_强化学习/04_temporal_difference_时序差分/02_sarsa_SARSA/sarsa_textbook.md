# SARSA教材

## 第一章：SARSA简介

### 1.1 什么是SARSA

SARSA（State-Action-Reward-State-Action）是一种On-Policy的时序差分学习方法。

### 1.2 名称来源

SARSA = State-Action-Reward-State-Action，代表更新中包含的五个元素。

### 1.3 与Q学习的区别

| 特性 | Q学习 | SARSA |
|------|-------|-------|
| 策略类型 | Off-Policy | On-Policy |
| 更新目标 | max_a' Q(s',a') | Q(s',a') |

## 第二章：SARSA更新规则

### 2.1 更新公式

$$ Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \cdot Q(s',a') - Q(s,a)] $$

### 2.2 参数说明

- α：学习率
- γ：折扣因子
- a'：实际采取的下一个动作

## 第三章：算法流程

### 3.1 伪代码

```
1. 初始化 Q(s,a) = 0
2. 对每个episode:
   a. 初始化状态s, 选择动作a (ε-贪婪)
   b. 对每步:
      i. 执行动作a, 观察r, s'
      ii. 选择下一动作a' (ε-贪婪)
      iii. 更新 Q(s,a)
      iv. s = s', a = a'
   c. 直到终止状态
```

## 第四章：特点

### 4.1 优点

1. 安全探索
2. 适合连续任务
3. 稳定收敛

### 4.2 缺点

1. 可能不是最优
2. 收敛较慢

## 第五章：代码实现

```python
def sarsa(n_episodes, alpha=0.1, gamma=0.95, epsilon=0.1):
    Q = np.zeros((n_states, n_actions))
    
    for episode in range(n_episodes):
        state = start
        action = epsilon_greedy(Q[state], epsilon)
        
        while not done:
            next_state, reward, done = env.step(state, action)
            next_action = epsilon_greedy(Q[next_state], epsilon)
            
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            
            state = next_state
            action = next_action
    
    return Q
```

## 第六章：总结

SARSA是重要的On-Policy TD学习方法，适合需要安全探索的场景。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
