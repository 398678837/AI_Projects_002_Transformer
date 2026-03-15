"""
强化学习算法对比演示
对比不同算法的性能
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("强化学习算法对比演示")
print("=" * 70)

# 1. 定义简化的实验
print("\n1. 算法对比实验...")

class SimpleEnv:
    def __init__(self):
        self.n_states = 16
        self.n_actions = 4
        
    def reset(self):
        return 0
    
    def step(self, action):
        next_state = np.random.randint(self.n_states)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        return next_state, reward, done

env = SimpleEnv()

# 2. 模拟不同算法的学习曲线
print("\n2. 模拟各算法的学习曲线...")

def simulate_q_learning(n_episodes=200):
    """模拟Q学习"""
    rewards = []
    for i in range(n_episodes):
        r = -2 + 8 * (1 - np.exp(-i/30)) + np.random.randn() * 0.5
        rewards.append(r)
    return rewards

def simulate_sarsa(n_episodes=200):
    """模拟SARSA"""
    rewards = []
    for i in range(n_episodes):
        r = -2 + 7 * (1 - np.exp(-i/35)) + np.random.randn() * 0.5
        rewards.append(r)
    return rewards

def simulate_dqn(n_episodes=200):
    """模拟DQN"""
    rewards = []
    for i in range(n_episodes):
        r = -2 + 8.5 * (1 - np.exp(-i/25)) + np.random.randn() * 0.4
        rewards.append(r)
    return rewards

def simulate_ppo(n_episodes=200):
    """模拟PPO"""
    rewards = []
    for i in range(n_episodes):
        r = -2 + 9 * (1 - np.exp(-i/20)) + np.random.randn() * 0.3
        rewards.append(r)
    return rewards

def simulate_policy_gradient(n_episodes=200):
    """模拟策略梯度"""
    rewards = []
    for i in range(n_episodes):
        r = -2 + 6 * (1 - np.exp(-i/40)) + np.random.randn() * 0.8
        rewards.append(r)
    return rewards

# 生成模拟数据
np.random.seed(42)
data = {
    'Q-Learning': simulate_q_learning(),
    'SARSA': simulate_sarsa(),
    'DQN': simulate_dqn(),
    'PPO': simulate_ppo(),
    'Policy Gradient': simulate_policy_gradient()
}

# 3. 可视化
print("\n3. 可视化对比...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 学习曲线
ax = axes[0]
colors = ['blue', 'green', 'red', 'purple', 'orange']
for (name, rewards), color in zip(data.items(), colors):
    smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
    ax.plot(smoothed, label=name, color=color, linewidth=2)

ax.set_xlabel('Episode')
ax.set_ylabel('平均奖励')
ax.set_title('强化学习算法学习曲线对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 性能对比
ax = axes[1]
final_rewards = [np.mean(data[name][-50:]) for name in data.keys()]
bars = ax.bar(data.keys(), final_rewards, color=colors, alpha=0.7)
ax.set_ylabel('最终平均奖励')
ax.set_title('算法最终性能对比')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

for bar, reward in zip(bars, final_rewards):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
           f'{reward:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('images/algorithm_comparison.png')
print("可视化已保存为 'images/algorithm_comparison.png'")

# 4. 总结表
print("\n4. 算法对比总结...")

summary = """
| 算法 | 类型 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|----------|
| Q-Learning | Off-Policy | 收敛快 | 需要探索 | 离散动作 |
| SARSA | On-Policy | 安全 | 收敛慢 | 安全关键 |
| DQN | Off-Policy | 可处理高维 | 离散动作 | Atari游戏 |
| PPO | On-Policy | 稳定 | 样本效率 | 通用 |
| Actor-Critic | On-Policy | 方差低 | 复杂 | 连续动作 |
"""
print(summary)

# 5. 完整总结
print("\n" + "=" * 70)
print("算法对比总结")
print("=" * 70)
print("""
选择算法的建议:

1. 离散动作空间:
   - 简单问题: Q-Learning, SARSA
   - 复杂问题: DQN, Double DQN

2. 连续动作空间:
   - 简单问题: Policy Gradient
   - 复杂问题: PPO, Actor-Critic

3. 需要安全探索:
   - 选择SARSA或On-Policy方法

4. 样本效率:
   - Off-Policy方法更好 (DQN, SAC)

5. 稳定收敛:
   - PPO是目前最稳定的选择
""")
print("=" * 70)
print("\nAlgorithm Comparison Demo完成！")
