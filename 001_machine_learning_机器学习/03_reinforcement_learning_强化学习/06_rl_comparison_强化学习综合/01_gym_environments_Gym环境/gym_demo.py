"""
Gym环境演示
使用OpenAI Gym环境展示各种强化学习任务
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Gym环境演示")
print("=" * 70)

# 1. 定义几个经典环境
print("\n1. 经典强化学习环境...")

class GridWorld:
    """简化GridWorld环境"""
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.state = self.start
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        i, j = self.state
        if action == 0: i = max(0, i-1)
        elif action == 1: i = min(self.size-1, i+1)
        elif action == 2: j = max(0, j-1)
        elif action == 3: j = min(self.size-1, j+1)
        
        self.state = (i, j)
        
        if self.state == self.goal:
            return self.state, 10, True
        return self.state, -1, False

class CartPole:
    """简化CartPole环境"""
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        
    def reset(self):
        self.state = np.random.randn(4) * 0.1
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = 10.0 if action == 1 else -10.0
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (force + 0.1 * 0.5 * theta_dot**2 * sintheta) / 1.1
        thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0/3.0 - 0.1 * costheta**2 / 1.1))
        xacc = temp - 0.1 * thetaacc * costheta / 1.1
        x, x_dot = x + 0.02*x_dot, x_dot + 0.02*xacc
        theta, theta_dot = theta + 0.02*theta_dot, theta_dot + 0.02*thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        done = abs(x) > 2.4 or abs(theta) > 0.2095
        return self.state, 1.0 if not done else 0.0, done

class MountainCar:
    """简化MountainCar环境"""
    def __init__(self):
        self.action_space = 3
        
    def reset(self):
        self.state = np.array([-0.5, 0.0])
        return self.state
    
    def step(self, action):
        position, velocity = self.state
        velocity += (action - 1) * 0.001 + np.cos(3*position) * -0.0025
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        position = np.clip(position, -1.2, 0.6)
        
        if position >= 0.6:
            return np.array([position, velocity]), 0, True
        return np.array([position, velocity]), -1, False

# 2. 环境对比
print("\n2. 环境对比...")

envs = {
    'GridWorld': GridWorld(5),
    'CartPole': CartPole(),
    'MountainCar': MountainCar()
}

print("\n环境信息:")
for name, env in envs.items():
    if hasattr(env, 'n_states'):
        states = env.n_states
    else:
        states = '连续'
    if hasattr(env, 'n_actions'):
        actions = env.n_actions
    else:
        actions = env.action_space
    print(f"  {name}: 状态={states}, 动作={actions}")

# 3. 随机策略测试
print("\n3. 随机策略测试...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, env) in enumerate(envs.items()):
    rewards = []
    for episode in range(50):
        state = env.reset()
        total_reward = 0
        while True:
            action = np.random.randint(env.action_space if hasattr(env, 'action_space') else env.n_actions)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
    
    ax = axes[idx]
    ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'{name} - Random Policy')
    ax.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Avg: {np.mean(rewards):.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('images/gym_environments.png')
print("\n可视化已保存为 'images/gym_environments.png'")

# 4. 总结
print("\n" + "=" * 70)
print("Gym环境总结")
print("=" * 70)
print("""
经典强化学习环境:

1. GridWorld (格子世界):
   - 离散状态和动作
   - 简单直观
   - 适合教学

2. CartPole (倒立摆):
   - 连续状态空间
   - 离散动作空间
   - 经典控制问题

3. MountainCar (山地车):
   - 连续状态和动作
   - 需要探索才能到达目标
   - 困难任务

常用环境库:
- OpenAI Gym
- Gymnasium
- DeepMind Control Suite
- MuJoCo
""")
print("=" * 70)
print("\nGym Environment Demo完成！")
