# Actor-Critic详细文档

## 1. 概念介绍

### 1.1 什么是Actor-Critic

Actor-Critic（演员-评论家）是一种结合了价值函数和策略梯度的强化学习算法框架。它由两个组件组成：
- **Actor（演员）**：负责学习策略
- **Critic（评论家）**：负责评估价值函数

### 1.2 为什么需要Actor-Critic

- **纯策略梯度（REINFORCE）**：方差高，收敛慢
- **纯价值函数（Q学习）**：不适合连续动作
- **Actor-Critic**：结合两者优点，方差低，收敛快

## 2. 技术原理

### 2.1 核心思想

使用Critic估计的价值函数来减少策略梯度的方差：
$$ \nabla_\theta J(\theta) \approx \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)] $$

其中$A(s,a)$是优势函数。

### 2.2 优势函数

$$ A(s,a) = Q(s,a) - V(s) $$

简化版本使用TD误差：
$$ A(s,a) \approx \delta = r + \gamma V(s') - V(s) $$

### 2.3 算法流程

```
1. 初始化Actor和Critic网络
2. 对每个episode:
   a. 使用当前策略采样
   b. 计算TD误差
   c. 更新Critic: 最小化TD误差
   d. 更新Actor: 策略梯度 * TD误差
```

## 3. 代码实现

```python
def actor_critic():
    for episode in range(n_episodes):
        state = env.reset()
        
        while not done:
            action = actor.sample(state)
            next_state, reward, done = env.step(action)
            
            td_error = reward + gamma * critic(next_state) - critic(state)
            
            critic.update(td_error)
            actor.update(td_error * log_prob)
            
            state = next_state
```

## 4. 优缺点

### 优点
- 方差低
- 收敛快
- 适合连续动作

### 缺点
- 实现复杂
- 需要平衡两个网络

## 5. 总结

Actor-Critic是现代强化学习的基础框架。
