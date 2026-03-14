# PPO（Proximal Policy Optimization）详细文档

## 1. 概念介绍

### 1.1 什么是PPO

PPO（近端策略优化）由Schulman等人于2017年提出，是目前最流行的深度强化学习算法之一。它通过限制策略更新的幅度来提高训练的稳定性和样本效率。

### 1.2 PPO的核心思想

PPO的核心是**信赖域优化**和**裁剪目标函数**，它避免策略在一次更新中发生剧烈变化。

### 1.3 PPO vs 其他算法

| 特性 | PPO | A2C | DQN |
|------|-----|-----|-----|
| 策略类型 | On-Policy | On-Policy | Off-Policy |
| 样本效率 | 中等 | 中等 | 高 |
| 稳定性 | 高 | 中等 | 中等 |

## 2. 技术原理

### 2.1 裁剪目标函数

$$ L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)] $$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon$ 是裁剪超参数（通常为0.2）

### 2.2 GAE（广义优势估计）

$$ A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} $$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### 2.3 PPO算法流程

```
1. 收集数据
2. 计算GAE
3. 多次更新:
   - 计算裁剪损失
   - 更新策略网络
   - 更新价值网络
4. 重复
```

## 3. 代码实现

```python
def ppo_update():
    for _ in range(k_epochs):
        for t in range(steps_per_batch):
            ratio = exp(new_log_prob - old_log_prob)
            clipped_ratio = clip(ratio, 1-eps, 1+eps)
            loss = -min(ratio * advantage, clipped_ratio * advantage)
            optimizer.step()
```

## 4. 优缺点

### 优点
- 稳定收敛
- 样本效率高
- 超参数友好

### 缺点
- On-Policy，样本利用率有限
- 需要较多调参

## 5. 总结

PPO是目前最成功的深度强化学习算法之一。
