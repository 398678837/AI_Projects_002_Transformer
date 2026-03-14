# Gym环境教材

## 第一章：Gym简介

### 1.1 什么是Gym

OpenAI Gym是强化学习环境库。

## 第二章：经典环境

### 2.1 离散环境

- GridWorld
- FrozenLake

### 2.2 控制环境

- CartPole
- MountainCar

## 第三章：接口

```python
env = gym.make('CartPole-v1')
state = env.reset()
next_state, reward, done, info = env.step(action)
```

## 第四章：总结

Gym是强化学习基础平台。
