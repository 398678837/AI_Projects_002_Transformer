"""
Actor-Critic演示
结合价值函数的策略梯度方法
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Actor-Critic演示")
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

# 2. Actor-Critic网络
print("\n2. 定义Actor-Critic网络...")

class ActorCritic:
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
    
    def forward_actor(self, state):
        hidden = np.tanh(np.dot(state, self.actor_W1) + self.actor_b1)
        logits = np.dot(hidden, self.actor_W2) + self.actor_b2
        probs = self.softmax(logits)
        return probs, hidden
    
    def forward_critic(self, state):
        hidden = np.tanh(np.dot(state, self.critic_W1) + self.critic_b1)
        value = np.dot(hidden, self.critic_W2) + self.critic_b2
        return value, hidden
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / (exp_x.sum() + 1e-8)
    
    def sample_action(self, state):
        probs, _ = self.forward_actor(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs[action]
    
    def get_value(self, state):
        value, _ = self.forward_critic(state)
        return value[0]

# 3. Actor-Critic算法
print("\n3. Actor-Critic算法...")

def actor_critic(n_episodes=200, gamma=0.99, actor_lr=0.01, critic_lr=0.1):
    """Actor-Critic算法"""
    agent = ActorCritic(env.state_space, env.action_space)
    rewards_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action, prob = agent.sample_action(state)
            next_state, reward, done = agent.forward_critic(state)[0][0]
            
            # 计算TD目标
            if done:
                target = reward
            else:
                next_value = agent.get_value(next_state)
                target = reward + gamma * next_value
            
            # 计算TD误差 (advantage)
            td_error = target - agent.get_value(state)
            
            # Critic更新
            _, critic_hidden = agent.forward_critic(state)
            agent.critic_W2 += critic_lr * td_error * critic_hidden
            agent.critic_b2 += critic_lr * td_error
            
            # Actor更新 (使用TD误差作为advantage)
            _, actor_hidden = agent.forward_actor(state)
            log_prob = np.log(prob + 1e-8)
            agent.actor_W2[action] += actor_lr * td_error * actor_hidden
            agent.actor_b2[action] += actor_lr * td_error
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"  Episode {episode + 1}, 平均奖励: {avg_reward:.1f}")
    
    return rewards_history, agent

rewards_history, agent = actor_critic(n_episodes=200)

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
ax.set_title('Actor-Critic学习曲线 (CartPole)')
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.legend()

ax = axes[1]
test_rewards = []
for _ in range(10):
    state = env.reset()
    total = 0
    while True:
        action, _ = agent.sample_action(state)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break
    test_rewards.append(total)

ax.bar(['Actor-Critic'], [np.mean(test_rewards)], yerr=np.std(test_rewards), color='purple', alpha=0.7)
ax.axhline(y=195, color='r', linestyle='--', label='成功阈值')
ax.set_ylabel('平均奖励')
ax.set_title('策略评估')
ax.legend()

plt.tight_layout()
plt.savefig('images/actor_critic_result.png')
print("可视化已保存为 'images/actor_critic_result.png'")

# 5. 总结
print("\n" + "=" * 70)
print("Actor-Critic总结")
print("=" * 70)
print("""
Actor-Critic核心思想:

1. 结合价值函数与策略梯度:
   - Actor: 策略网络，学习策略
   - Critic: 价值网络，评估状态

2. 优势函数 (Advantage):
   A(s,a) = Q(s,a) - V(s)
   减少策略梯度方差

3. 更新规则:
   - Critic: 最小化TD误差
   - Actor: 最大化期望回报

4. 优点:
   - 方差低
   - 收敛快
   - 适合连续任务
""")
print("=" * 70)
print("\nActor-Critic Demo完成！")
