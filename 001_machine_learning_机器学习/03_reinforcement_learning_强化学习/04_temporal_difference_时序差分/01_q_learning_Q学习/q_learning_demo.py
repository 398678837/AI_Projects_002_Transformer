"""
Q学习（Q-Learning）演示
使用格子世界环境展示Q学习算法
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("Q学习（Q-Learning）演示")
print("=" * 70)

# 1. 定义环境
print("\n1. 定义格子世界环境...")

grid_size = 5
n_states = grid_size * grid_size
n_actions = 4  # 上、下、左、右

actions = ['up', 'down', 'left', 'right']
action_to_idx = {a: i for i, a in enumerate(actions)}

# 目标位置和陷阱
goal = (4, 4)
trap = (2, 2)
start = (0, 0)

# 奖励
reward_goal = 10
reward_trap = -10
reward_step = -1

print(f"格子大小: {grid_size}x{grid_size}")
print(f"状态数量: {n_states}")
print(f"动作数量: {n_actions}")
print(f"目标位置: {goal}, 奖励: {reward_goal}")
print(f"陷阱位置: {trap}, 奖励: {reward_trap}")

# 2. 状态转移和奖励
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
    else:
        return reward_step

# 3. Q学习算法
print("\n2. Q学习算法...")

def q_learning(n_episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Q学习算法"""
    Q = np.zeros((n_states, n_actions))
    
    state_to_idx = lambda s: s[0] * grid_size + s[1]
    
    rewards_history = []
    
    for episode in range(n_episodes):
        state = start
        state_idx = state_to_idx(state)
        total_reward = 0
        
        while True:
            # ε-贪婪策略
            if np.random.random() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                action_idx = np.argmax(Q[state_idx])
            
            # 执行动作
            next_state = get_next_state(state, actions[action_idx])
            reward = get_reward(next_state)
            next_state_idx = state_to_idx(next_state)
            
            total_reward += reward
            
            # Q学习更新
            if next_state == goal or next_state == trap:
                Q[state_idx, action_idx] += alpha * (reward - Q[state_idx, action_idx])
            else:
                Q[state_idx, action_idx] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action_idx])
            
            state = next_state
            state_idx = next_state_idx
            
            # 到达终止状态
            if next_state == goal or next_state == trap:
                break
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"  Episode {episode + 1}, 平均奖励: {avg_reward:.2f}")
    
    return Q, rewards_history

Q, rewards_history = q_learning(n_episodes=500)

# 4. 提取策略
print("\n3. 提取最优策略...")

state_to_idx = lambda s: s[0] * grid_size + s[1]

def extract_policy(Q):
    """从Q函数提取策略"""
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

print("最优策略:")
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
        else:
            row += ". "
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
ax.set_title('Q学习学习曲线')
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

ax.set_title('Q学习: 状态价值 & 最优策略')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/q_learning_result.png')
print("可视化已保存为 'images/q_learning_result.png'")

# 6. 总结
print("\n" + "=" * 70)
print("Q学习总结")
print("=" * 70)
print("""
Q学习算法 (Off-Policy TD Control):

更新规则:
  Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]

特点:
1. Off-Policy: 学习最优策略而不需要遵循它
2. TD学习: 结合了蒙特卡洛和动态规划的优点
3. 异步更新: 不需要等待episode结束
4. 收敛性: 在适当条件下保证收敛到最优Q函数

核心思想:
- 直接学习最优动作价值函数
- 使用贪婪策略选择动作
""")
print("=" * 70)
print("\nQ-Learning Demo完成！")
