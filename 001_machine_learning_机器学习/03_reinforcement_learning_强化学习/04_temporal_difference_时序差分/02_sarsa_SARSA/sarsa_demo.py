"""
SARSA演示
使用格子世界展示SARSA算法（On-Policy TD Control）
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("SARSA演示")
print("=" * 70)

# 1. 定义环境
print("\n1. 定义格子世界环境...")

grid_size = 5
n_states = grid_size * grid_size
n_actions = 4

actions = ['up', 'down', 'left', 'right']
action_to_idx = {a: i for i, a in enumerate(actions)}

goal = (4, 4)
trap = (2, 2)
start = (0, 0)

reward_goal = 10
reward_trap = -10
reward_step = -1

print(f"格子大小: {grid_size}x{grid_size}")
print(f"目标: {goal}, 陷阱: {trap}")

# 2. 环境函数
def get_next_state(state, action):
    i, j = state
    if action == 'up':
        i = max(0, i - 1)
    elif action == 'down':
        i = min(grid_size - 1, i + 1)
    elif action == 'left':
        j = max(0, j - 1)
    elif action == 'right':
        j = min(grid_size - 1, j + 1)
    return (i, j)

def get_reward(next_state):
    if next_state == goal:
        return reward_goal
    elif next_state == trap:
        return reward_trap
    return reward_step

# 3. SARSA算法
print("\n2. SARSA算法...")

def sarsa(n_episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1):
    """SARSA算法 - On-Policy TD Control"""
    Q = np.zeros((n_states, n_actions))
    
    state_to_idx = lambda s: s[0] * grid_size + s[1]
    
    rewards_history = []
    
    for episode in range(n_episodes):
        # 衰减epsilon
        current_epsilon = epsilon * (1 - episode / n_episodes)
        
        state = start
        state_idx = state_to_idx(state)
        
        # 初始动作选择（使用epsilon-贪婪）
        if np.random.random() < current_epsilon:
            action_idx = np.random.randint(n_actions)
        else:
            action_idx = np.argmax(Q[state_idx])
        
        total_reward = 0
        
        while True:
            # 执行动作
            next_state = get_next_state(state, actions[action_idx])
            reward = get_reward(next_state)
            next_state_idx = state_to_idx(next_state)
            
            total_reward += reward
            
            # 选择下一个动作（使用epsilon-贪婪）
            if np.random.random() < current_epsilon:
                next_action_idx = np.random.randint(n_actions)
            else:
                next_action_idx = np.argmax(Q[next_state_idx])
            
            # SARSA更新（使用实际的下一动作）
            if next_state == goal or next_state == trap:
                Q[state_idx, action_idx] += alpha * (reward - Q[state_idx, action_idx])
            else:
                Q[state_idx, action_idx] += alpha * (reward + gamma * Q[next_state_idx, next_action_idx] - Q[state_idx, action_idx])
            
            state = next_state
            state_idx = next_state_idx
            action_idx = next_action_idx
            
            if next_state == goal or next_state == trap:
                break
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"  Episode {episode + 1}, epsilon={current_epsilon:.3f}, 平均奖励: {avg_reward:.2f}")
    
    return Q, rewards_history

Q, rewards_history = sarsa(n_episodes=500)

# 4. 提取策略
print("\n3. 提取策略...")

state_to_idx = lambda s: s[0] * grid_size + s[1]

def extract_policy(Q):
    policy = {}
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            if state not in [goal, trap]:
                state_idx = state_to_idx(state)
                action_idx = np.argmax(Q[state_idx])
                policy[state] = actions[action_idx]
    return policy

optimal_policy = extract_policy(Q)

print("SARSA学习到的策略:")
for i in range(grid_size):
    row = ""
    for j in range(grid_size):
        state = (i, j)
        if state == goal:
            row += "G "
        elif state == trap:
            row += "X "
        elif state in optimal_policy:
            arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}[optimal_policy[state]]
            row += arrow + " "
    print(row)

# 5. 可视化
print("\n4. 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 学习曲线
ax = axes[0]
window = 20
rewards_smoothed = []
for i in range(0, len(rewards_history), window):
    rewards_smoothed.append(np.mean(rewards_history[i:i+window]))

ax.plot(range(window, len(rewards_history) + 1, window), rewards_smoothed)
ax.set_xlabel('Episode')
ax.set_ylabel('总奖励')
ax.set_title('SARSA学习曲线')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# Q值热力图
ax = axes[1]
V = np.max(Q, axis=1).reshape(grid_size, grid_size)
V[goal[0], goal[1]] = reward_goal
V[trap[0], trap[1]] = reward_trap

im = ax.imshow(V, cmap='RdYlGn', origin='lower')
for i in range(grid_size):
    for j in range(grid_size):
        state = (i, j)
        if state == goal:
            ax.text(j, i, 'G', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        elif state == trap:
            ax.text(j, i, 'X', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
        else:
            arrow = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}.get(optimal_policy.get(state, ''), '')
            ax.text(j, i, f'{V[i,j]:.1f}\n{arrow}', ha='center', va='center', fontsize=9, color='white')

ax.set_title('SARSA: 状态价值 & 策略')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/sarsa_result.png')
print("可视化已保存为 'images/sarsa_result.png'")

# 6. 总结
print("\n" + "=" * 70)
print("SARSA总结")
print("=" * 70)
print("""
SARSA算法 (On-Policy TD Control):

更新规则:
  Q(s,a) ← Q(s,a) + α[r + γ * Q(s',a') - Q(s,a)]
  
  其中 a' 是实际执行的下一个动作

与Q学习的区别:
- Q学习: Off-Policy, 学习max_a' Q(s',a')
- SARSA: On-Policy, 学习实际采取的Q(s',a')

SARSA特点:
1. On-Policy: 评估和改进同一策略
2. 更安全的探索: 考虑了探索的风险
3. 适合连续任务: 不会因探索而崩溃
4. 收敛更稳定: 探索更保守
""")
print("=" * 70)
print("\nSARSA Demo完成！")
