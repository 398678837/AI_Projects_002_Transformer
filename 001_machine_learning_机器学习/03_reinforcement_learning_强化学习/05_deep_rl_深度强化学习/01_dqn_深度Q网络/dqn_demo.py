"""
深度Q网络（Deep Q-Network, DQN）演示
使用CartPole环境展示DQN算法
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("深度Q网络（DQN）演示")
print("=" * 70)

# 1. 简化的CartPole环境
print("\n1. 定义简化CartPole环境...")

class SimpleCartPole:
    """简化的CartPole环境"""
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        
    def reset(self):
        self.state = np.random.randn(4) * 0.1
        self.steps = 0
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.mass_pole * self.length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / self.total_mass))
        xacc = temp - self.mass_pole * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        done = x < -2.4 or x > 2.4 or theta < -0.2095 or theta > 0.2095 or self.steps >= 500
        reward = 1.0 if not done else 0.0
        
        return self.state, reward, done

env = SimpleCartPole()
print(f"状态空间: {env.state_space}")
print(f"动作空间: {env.action_space}")

# 2. 简化神经网络
print("\n2. 定义Q网络（简化版）...")

class SimpleQNetwork:
    """简化的Q网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
    def forward(self, state):
        self.state = state
        self.hidden = np.tanh(np.dot(state, self.W1) + self.b1)
        self.output = np.dot(self.hidden, self.W2) + self.b2
        return self.output
    
    def get_q_value(self, state, action):
        q_values = self.forward(state)
        return q_values[action]
    
    def predict(self, state):
        return np.argmax(self.forward(state))

# 3. DQN算法
print("\n3. DQN算法...")

def dqn(n_episodes=200, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, 
        target_update_freq=10, memory_size=1000, batch_size=32):
    """DQN算法"""
    state_dim = env.state_space
    action_dim = env.action_space
    
    # Q网络和目标网络
    q_network = SimpleQNetwork(state_dim, action_dim)
    target_network = SimpleQNetwork(state_dim, action_dim)
    
    # 经验回放缓冲区
    memory = []
    memory_ptr = 0
    
    rewards_history = []
    q_values_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        q_sum = 0
        
        while True:
            # ε-贪婪策略
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                action = q_network.predict(state)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验
            memory.append((state, action, reward, next_state, done))
            if len(memory) > memory_size:
                memory.pop(0)
            
            # 经验回放
            if len(memory) >= batch_size:
                batch = np.random.choice(len(memory), batch_size, replace=False)
                for idx in batch:
                    s, a, r, s_next, d = memory[idx]
                    
                    if d:
                        target = r
                    else:
                        target = r + gamma * np.max(target_network.forward(s_next))
                    
                    # 简化的梯度更新
                    q_values = q_network.forward(s)
                    q_current = q_values[a]
                    error = target - q_current
                    
                    # 梯度更新
                    grad = error * 0.001
                    q_network.hidden = np.tanh(np.dot(s, q_network.W1) + q_network.b1)
                    q_network.W2[a] += grad * q_network.hidden
                    q_network.b2[a] += grad
            
            total_reward += reward
            q_sum += q_network.get_q_value(state, action)
            
            if done:
                break
            
            state = next_state
        
        # 更新目标网络
        if (episode + 1) % target_update_freq == 0:
            target_network.W1 = q_network.W1.copy()
            target_network.b1 = q_network.b1.copy()
            target_network.W2 = q_network.W2.copy()
            target_network.b2 = q_network.b2.copy()
        
        # 衰减epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        rewards_history.append(total_reward)
        q_values_history.append(q_sum / max(1, total_reward))
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"  Episode {episode + 1}, epsilon={epsilon:.3f}, 平均奖励: {avg_reward:.1f}")
    
    return rewards_history, q_network

rewards_history, q_network = dqn(n_episodes=200)

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 学习曲线
ax = axes[0]
window = 10
rewards_smoothed = []
for i in range(0, len(rewards_history), window):
    rewards_smoothed.append(np.mean(rewards_history[i:i+window]))

ax.plot(range(window, len(rewards_history) + 1, window), rewards_smoothed)
ax.set_xlabel('Episode')
ax.set_ylabel('总奖励')
ax.set_title('DQN学习曲线 (CartPole)')
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.legend()

# 策略评估
ax = axes[1]
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    while True:
        action = q_network.predict(state)
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    test_rewards.append(total_reward)

ax.bar(['DQN策略'], [np.mean(test_rewards)], yerr=np.std(test_rewards), color='blue', alpha=0.7)
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.set_ylabel('平均奖励')
ax.set_title('DQN策略评估')
ax.legend()

plt.tight_layout()
plt.savefig('images/dqn_result.png')
print("可视化已保存为 'images/dqn_result.png'")

# 5. 总结
print("\n" + "=" * 70)
print("DQN总结")
print("=" * 70)
print("""
深度Q网络 (DQN) 核心思想:

1. 深度神经网络逼近Q函数:
   Q(s,a;θ) ≈ Q*(s,a)

2. 经验回放 (Experience Replay):
   - 存储 (s,a,r,s',d) 到回放缓冲区
   - 随机采样进行学习，打破数据相关性

3. 目标网络 (Target Network):
   - 固定目标网络，定期更新
   - 稳定训练，减少训练震荡

4. 损失函数:
   L(θ) = [r + γ * max_a' Q_target(s',a';θ-) - Q(s,a;θ)]²
""")
print("=" * 70)
print("\nDQN Demo完成！")
