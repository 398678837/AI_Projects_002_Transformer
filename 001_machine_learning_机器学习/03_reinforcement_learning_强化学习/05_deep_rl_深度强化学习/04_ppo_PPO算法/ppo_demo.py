"""
PPO（Proximal Policy Optimization）演示
信赖域策略优化算法
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("PPO演示")
print("=" * 70)

# 1. 环境
print("\n1. 定义CartPole环境...")

class SimpleCartPole:
    def __init__(self):
        self.state_space = 4
        self.action_space = 2
        
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
        return self.state, 1.0 if not done else 0.0, done

env = SimpleCartPole()

# 2. PPO网络
print("\n2. 定义PPO网络...")

class PPONetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        np.random.seed(42)
        self.actor_W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.actor_b2 = np.zeros(action_dim)
        
        self.critic_W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.critic_b1 = np.zeros(hidden_dim)
        self.critic_W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.critic_b2 = np.zeros(1)
        
        self.log_stds = np.zeros(action_dim)
        
    def forward_actor(self, state):
        hidden = np.tanh(np.dot(state, self.actor_W1) + self.actor_b1)
        logits = np.dot(hidden, self.actor_W2) + self.actor_b2
        probs = self.softmax(logits)
        return probs, hidden
    
    def forward_critic(self, state):
        hidden = np.tanh(np.dot(state, self.critic_W1) + self.critic_b1)
        value = np.dot(hidden, self.critic_W2) + self.critic_b2
        return value[0], hidden
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-8)
    
    def sample_action(self, state):
        probs, _ = self.forward_actor(state)
        action = np.random.choice(len(probs), p=probs)
        return action
    
    def get_log_prob(self, state, action):
        probs, _ = self.forward_actor(state)
        return np.log(probs[action] + 1e-8)

# 3. PPO算法
print("\n3. PPO算法...")

def ppo(n_episodes=200, gamma=0.99, lam=0.95, actor_lr=0.0003, critic_lr=0.001, 
        clip_epsilon=0.2, k_epochs=4):
    """PPO算法"""
    agent = PPONetwork(env.state_space, env.action_space)
    rewards_history = []
    
    for episode in range(n_episodes):
        # 收集轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        
        state = env.reset()
        
        while True:
            action = agent.sample_action(state)
            next_state, reward, done = agent.forward_critic(state)[0]
            
            log_prob = agent.get_log_prob(state, action)
            value = agent.get_value(state)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            if done:
                break
            state = next_state
        
        total_reward = len(rewards)
        rewards_history.append(total_reward)
        
        # 计算GAE
        advantages = []
        gae = 0
        for t in range(len(rewards) - 1, -1, -1):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(values)
        
        # PPO更新
        for _ in range(k_epochs):
            for t in range(len(states)):
                state = states[t]
                action = actions[t]
                old_log_prob = log_probs[t]
                advantage = advantages[t]
                
                # 计算新的log_prob和value
                new_log_prob = agent.get_log_prob(state, action)
                new_value, _ = agent.forward_critic(state)
                
                # PPO损失 (简化版)
                ratio = np.exp(new_log_prob - old_log_prob)
                clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                
                # Actor更新
                actor_loss = -min(ratio * advantage, clipped_ratio * advantage)
                _, actor_hidden = agent.forward_actor(state)
                agent.actor_W2[action] += actor_lr * actor_loss * actor_hidden
                
                # Critic更新
                critic_loss = (returns[t] - new_value) ** 2
                _, critic_hidden = agent.forward_critic(state)
                agent.critic_W2 += critic_lr * critic_loss * critic_hidden
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"  Episode {episode + 1}, 平均奖励: {avg_reward:.1f}")
    
    return rewards_history, agent

rewards_history, agent = ppo(n_episodes=200)

# 4. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
window = 10
rewards_smoothed = []
for i in range(0, len(rewards_history), window):
    rewards_smoothed.append(np.mean(rewards_history[i:i+window]))

ax.plot(range(window, len(rewards_history) + 1, window), rewards_smoothed)
ax.set_xlabel('Episode')
ax.set_ylabel('总奖励')
ax.set_title('PPO学习曲线 (CartPole)')
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.legend()

ax = axes[1]
test_rewards = []
for _ in range(10):
    state = env.reset()
    total = 0
    while True:
        action = agent.sample_action(state)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    test_rewards.append(total)

ax.bar(['PPO'], [np.mean(test_rewards)], yerr=np.std(test_rewards), color='orange', alpha=0.7)
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.set_ylabel('平均奖励')
ax.set_title('策略评估')
ax.legend()

plt.tight_layout()
plt.savefig('images/ppo_result.png')
print("可视化已保存为 'images/ppo_result.png'")

# 5. 总结
print("\n" + "=" * 70)
print("PPO总结")
print("=" * 70)
print("""
PPO (Proximal Policy Optimization) 核心思想:

1. 信赖域优化:
   - 限制策略更新的幅度
   - 避免策略剧烈变化

2. 裁剪目标函数:
   L^CLIP(θ) = E[min(r(θ)A, clip(r(θ),1-ε,1+ε)A)]

3. 特点:
   - 稳定收敛
   - 样本效率高
   - 超参数友好

4. 广泛使用:
   - 游戏AI
   - 机器人控制
   - 大规模训练
""")
print("=" * 70)
print("\nPPO Demo完成！")
