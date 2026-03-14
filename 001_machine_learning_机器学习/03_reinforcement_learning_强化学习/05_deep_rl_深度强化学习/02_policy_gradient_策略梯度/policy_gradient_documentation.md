# 策略梯度（Policy Gradient）详细文档

## 1. 概念介绍

### 1.1 什么是策略梯度

策略梯度是强化学习中一类直接优化策略参数的算法家族。与基于价值的方法（如Q学习）不同，策略梯度方法直接学习一个从状态到动作概率分布的映射。

### 1.2 策略梯度的核心思想

策略梯度方法直接计算目标函数相对于策略参数的梯度，然后使用梯度上升法更新参数。

### 1.3 策略梯度 vs Q学习

| 特性 | 策略梯度 | Q学习 |
|------|----------|-------|
| 表示 | 策略函数 π(a\|s) | 动作价值函数 Q(s,a) |
| 动作空间 | 连续或离散 | 离散 |
| 策略类型 | 随机策略 | 确定性策略 |
| 收敛性 | 局部最优 | 全局最优（tabular） |

## 2. 技术原理

### 2.1 策略梯度定理

策略梯度定理是策略梯度方法的基础：

$$ \nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a|s) \cdot G_t] $$

其中：
- $J(\theta)$：目标函数（平均回报）
- $\pi_\theta(a|s)$：策略函数
- $G_t$：回报

### 2.2 REINFORCE算法

REINFORCE是最基础的策略梯度算法：

```
1. 使用当前策略采样轨迹
2. 对每个时间步:
   计算 G_t
   ∇θ += ∇θ log πθ(a|s) * G_t
3. 更新参数: θ += α * ∴
```

### 2.3 带基线的REINFORCE

为了减少方差，可以引入基线：

$$ \nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a|s) \cdot (G_t - b(s))] $$

常用的基线包括：
- 状态价值函数 V(s)
- 平均回报

## 3. 代码实现

### 3.1 核心代码

```python
def reinforce(policy, env, n_episodes, gamma=0.99, lr=0.01):
    for episode in range(n_episodes):
        trajectory = []
        state = env.reset()
        
        while True:
            action = policy.sample_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward))
            if done:
                break
            state = next_state
        
        G = 0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward = trajectory[t]
            G = gamma * G + reward
            
            log_prob = policy.get_log_prob(state, action)
            policy.update(log_prob * G)
```

## 4. 优缺点分析

### 4.1 优点

1. **可以直接学习随机策略**
2. **适合连续动作空间**
3. **收敛性质好**
4. **可以学习复杂的策略**

### 4.2 缺点

1. **方差较高**
2. **收敛到局部最优**
3. **样本效率低**

## 5. 改进算法

- **Actor-Critic**：结合价值函数减少方差
- **PPO**：信赖域方法，更稳定
- **A2C/A3C**：异步方法，更快

## 6. 总结

策略梯度是强化学习的重要组成部分，特别适合连续动作空间和需要随机策略的场景。

---

**参考资料**：
- Williams, R.J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"
- Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning: An Introduction"
