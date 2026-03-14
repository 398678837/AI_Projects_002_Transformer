"""
策略梯度（Policy Gradient）演示
使用CartPole环境展示REINFORCE算法
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("策略梯度（Policy Gradient）演示")
print("=" * 70)

# 1. 环境定义
print("\n1. 定义CartPole环境...")

class SimpleCartPole:
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        self.gravity = 9.8
        
    def reset(self):
        self.state = np.random.randn(4) * 0.1
        self.steps = 0
        return self.state
    
    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        force = 10.0 if action == 1 else -10.0
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + 0.1 * 0.5 * theta_dot**2 * sintheta) / 1.1
        thetaacc = (9.8 * sintheta - costheta * temp) / (0.5 * (4.0/3.0 - 0.1 * costheta**2 / 1.1))
        xacc = temp - 0.1 * thetaacc * costheta / 1.1
        
        x = x + 0.02 * x_dot
        x_dot = x_dot + 0.02 * xacc
        theta = theta + 0.02 * theta_dot
        theta_dot = theta_dot + 0.02 * thetaacc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        done = abs(x) > 2.4 or abs(theta) > 0.2095 or self.steps >= 500
        reward = 1.0 if not done else 0.0
        
        return self.state, reward, done

env = SimpleCartPole()

# 2. 策略网络
print("\n2. 定义策略网络...")

class PolicyNetwork:
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
    def forward(self, state):
        self.hidden = np.tanh(np.dot(state, self.W1) + self.b1)
        logits = np.dot(self.hidden, self.W2) + self.b2
        self.probs = self.softmax(logits)
        return self.probs
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-8)
    
    def sample_action(self, state):
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]
    
    def get_log_prob(self, state, action):
        probs = self.forward(state)
        return np.log(probs[action] + 1e-8)

# 3. REINFORCE算法
print("\n3. REINFORCE算法...")

def reinforce(n_episodes=200, gamma=0.99, lr=0.01):
    """REINFORCE算法"""
    policy = PolicyNetwork(env.state_space, env.action_space)
    rewards_history = []
    
    for episode in range(n_episodes):
        # 收集episode
        trajectory = []
        state = env.reset()
        
        while True:
            action, prob = policy.sample_action(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, action, reward, prob))
            if done:
                break
            state = next_state
        
        total_reward = len(trajectory)
        rewards_history.append(total_reward)
        
        # 计算回报
        G = 0
        for t in range(len(trajectory) - 1, -1, -1):
            state, action, reward, _ = trajectory[t]
            G = gamma * G + reward
            
            # 策略梯度更新
            log_prob = policy.get_log_prob(state, action)
            gradient = log_prob * G
            
            # 简化的梯度更新
            hidden = np.tanh(np.dot(state, policy.W1) + policy.b1)
            policy.W2[action] += lr * gradient * hidden
            policy.b2[action] += lr * gradient
            policy.W1 += lr * gradient * 0.01 * np.outer(state, hidden)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"  Episode {episode + 1}, 平均奖励: {avg_reward:.1f}")
    
    return rewards_history, policy

rewards_history, policy = reinforce(n_episodes=200)

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
ax.set_title('REINFORCE学习曲线 (CartPole)')
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.legend()

# 策略评估
ax = axes[1]
test_rewards = []
for _ in range(10):
    state = env.reset()
    total_reward = 0
    while True:
        action, _ = policy.sample_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
        if done:
            break
    test_rewards.append(total_reward)

ax.bar(['REINFORCE策略'], [np.mean(test_rewards)], yerr=np.std(test_rewards), color='green', alpha=0.7)
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.set_ylabel('平均奖励')
ax.set_title('策略梯度策略评估')
ax.legend()

plt.tight_layout()
plt.savefig('images/policy_gradient_result.png')
print("可视化已保存为 'images/policy_gradient_result.png'")

# 5. 总结
print("\n" + "=" * 70)
print("策略梯度总结")
print("=" * 70)
print("""
策略梯度 (Policy Gradient) 核心思想:

1. 直接优化策略:
   - 学习策略函数 π(a|s;θ)
   - 直接计算梯度并更新

2. 策略梯度定理:
   ∇θ J(θ) = E[∇θ log πθ(a|s) * Gt]

3. REINFORCE算法:
   - 采样轨迹
   - 估计回报
   - 更新策略参数

4. 优点:
   - 可以学习随机策略
   - 适合连续动作空间
   - 收敛到局部最优
""")
print("=" * 70)
print("\nPolicy Gradient Demo完成！")
