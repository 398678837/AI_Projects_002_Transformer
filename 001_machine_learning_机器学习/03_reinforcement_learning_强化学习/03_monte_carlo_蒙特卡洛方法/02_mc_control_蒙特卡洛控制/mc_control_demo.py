"""
蒙特卡洛控制（Monte Carlo Control）演示
使用21点游戏展示蒙特卡洛控制过程
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("蒙特卡洛控制演示")
print("=" * 70)

# 1. 环境定义
print("\n1. 定义21点环境...")

def get_card():
    return np.random.randint(1, 11)

def get_initial_state():
    player_sum = get_card() + get_card()
    dealer_showing = get_card()
    usable_ace = 1 if player_sum <= 11 else 0
    return (player_sum, dealer_showing, usable_ace)

def step(state, action):
    player_sum, dealer_showing, usable_ace = state
    
    if action == 1:  # 要牌
        new_card = get_card()
        player_sum += new_card
        if new_card == 1 and player_sum + 10 <= 21:
            player_sum += 10
            usable_ace = 1
        
        if player_sum > 21:
            return state, -1, True
        
        return (player_sum, dealer_showing, usable_ace), 0, False
    
    else:  # 停牌
        dealer_sum = dealer_showing + get_card() + get_card()
        
        while dealer_sum < 17:
            dealer_sum += get_card()
        
        if dealer_sum > 21 or player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        else:
            reward = 0
        
        return state, reward, True

def get_action(state, epsilon):
    """ε-贪婪策略"""
    player_sum, dealer_showing, usable_ace = state
    
    if np.random.random() < epsilon:
        return np.random.randint(0, 2)
    else:
        return 0 if player_sum >= 20 else 1

# 2. 蒙特卡洛控制 - epsilon-greedy
print("\n2. 蒙特卡洛控制算法...")

def mc_control(n_episodes=50000, epsilon=0.1, gamma=1.0):
    """蒙特卡洛控制算法 - on-policy"""
    Q = {}
    returns_count = {}
    epsilon_decay = 0.995
    min_epsilon = 0.01
    
    rewards_history = []
    
    for episode_idx in range(n_episodes):
        current_epsilon = max(min_epsilon, epsilon * (epsilon_decay ** episode_idx))
        
        state = get_initial_state()
        episode = []
        
        while True:
            action = get_action(state, current_epsilon)
            next_state, reward, done = step(state, action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state
        
        # 计算回报并更新Q函数
        G = 0
        states_actions = set()
        
        for i in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[i]
            G = gamma * G + reward
            
            if (state, action) not in states_actions:
                states_actions.add((state, action))
                
                if (state, action) not in Q:
                    Q[(state, action)] = 0
                    returns_count[(state, action)] = 0
                
                returns_count[(state, action)] += 1
                Q[(state, action)] += (G - Q[(state, action)]) / returns_count[(state, action)]
        
        # 记录奖励
        rewards_history.append(reward)
        
        if (episode_idx + 1) % 10000 == 0:
            avg_reward = np.mean(rewards_history[-10000:])
            print(f"  Episode {episode_idx + 1}, epsilon={current_epsilon:.3f}, 平均奖励={avg_reward:.3f}")
    
    return Q, rewards_history

Q, rewards_history = mc_control(n_episodes=50000, epsilon=0.2)

# 3. 提取策略
print("\n3. 提取最优策略...")

def extract_policy(Q):
    """从Q函数提取策略"""
    policy = {}
    
    for (state, action), q_value in Q.items():
        if state not in policy:
            policy[state] = (action, q_value)
        else:
            if q_value > policy[state][1]:
                policy[state] = (action, q_value)
    
    return policy

optimal_policy = extract_policy(Q)

print(f"策略覆盖了 {len(optimal_policy)} 个状态")

# 4. 评估学习曲线
print("\n4. 评估学习曲线...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 奖励历史
ax = axes[0]
window = 1000
rewards_smoothed = []
for i in range(0, len(rewards_history), window):
    rewards_smoothed.append(np.mean(rewards_history[i:i+window]))

ax.plot(range(window, len(rewards_history) + 1, window), rewards_smoothed)
ax.set_xlabel('Episode')
ax.set_ylabel('平均奖励')
ax.set_title('蒙特卡洛控制学习曲线')
ax.axhline(y=0, color='r', linestyle='--', label='零基准')
ax.legend()

# 价值函数可视化
ax = axes[1]
player_sums = range(12, 22)
dealer_showing = range(1, 11)

V_matrix = np.zeros((len(player_sums), len(dealer_showing)))
for i, ps in enumerate(player_sums):
    for j, ds in enumerate(dealer_showing):
        state = (ps, ds, 0)
        if state in optimal_policy:
            action, q_val = optimal_policy[state]
            V_matrix[i, j] = q_val

im = ax.imshow(V_matrix, cmap='RdYlGn', origin='lower', aspect='auto')
ax.set_xlabel('庄家显示')
ax.set_ylabel('玩家总点数')
ax.set_title('学习到的Q值 (无可用Ace)')
ax.set_xticks(range(len(dealer_showing)))
ax.set_xticklabels(dealer_showing)
ax.set_yticks(range(len(player_sums)))
ax.set_yticklabels(player_sums)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/mc_control_learning.png')
print("学习曲线已保存为 'images/mc_control_learning.png'")

# 5. 总结
print("\n" + "=" * 70)
print("蒙特卡洛控制总结")
print("=" * 70)
print("""
蒙特卡洛控制算法 (On-Policy MC Control):

1. 探索开始 (Exploring Starts):
   - 从所有状态-动作对开始探索

2. 策略改进:
   - 使用ε-贪婪策略平衡探索与利用

3. 策略评估:
   - 用蒙特卡洛方法估计Q函数

4. 特点:
   - 逐步衰减ε以平衡探索利用
   - GLIE (Greedy in the Limit with Infinite Exploration)
""")
print("=" * 70)
print("\nMonte Carlo Control Demo完成！")
