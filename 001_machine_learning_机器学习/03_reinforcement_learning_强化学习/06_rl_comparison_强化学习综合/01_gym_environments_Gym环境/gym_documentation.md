# Gym环境详细文档

## 1. 概念介绍

### 1.1 什么是Gym

OpenAI Gym是强化学习最常用的环境库，提供了标准化的测试平台。

### 1.2 Gym的作用

- 提供标准化的环境接口
- 方便算法对比
- 促进强化学习研究

## 2. 经典环境

### 2.1 离散环境

| 环境 | 状态空间 | 动作空间 | 难度 |
|------|----------|----------|------|
| GridWorld | 离散 | 离散 | 简单 |
| FrozenLake | 离散 | 离散 | 中等 |
| Blackjack | 离散 | 离散 | 中等 |

### 2.2 控制环境

| 环境 | 状态空间 | 动作空间 | 特点 |
|------|----------|----------|------|
| CartPole | 连续 | 离散 | 经典入门 |
| MountainCar | 连续 | 离散 | 需要探索 |
| Pendulum | 连续 | 连续 | 连续控制 |

### 2.3 Atari游戏

- 高维图像状态
- 复杂策略
- 端到端学习

## 3. 环境接口

```python
import gym

env = gym.make('CartPole-v1')
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
```

## 4. 总结

Gym是强化学习研究和实验的基础平台。
