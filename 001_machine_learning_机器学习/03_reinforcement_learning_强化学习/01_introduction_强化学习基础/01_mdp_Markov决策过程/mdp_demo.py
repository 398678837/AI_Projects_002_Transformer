"""
Markov决策过程（MDP）的演示
使用简单的网格世界示例
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

# 1. 定义简单的网格世界
print("定义简单的网格世界...")

# 4x4的网格世界
grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
n_states = len(states)

# 动作：上、下、左、右
actions = ['up', 'down', 'left', 'right']
n_actions = len(actions)

# 定义奖励
rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 10  # 目标位置，奖励+10
rewards[1, 1] = -10  # 陷阱位置，奖励-10

print(f"状态数量: {n_states}")
print(f"动作数量: {n_actions}")
print("奖励矩阵:")
print(rewards)

# 2. 定义状态转移函数
print("\n定义状态转移函数...")

def get_next_state(state, action):
    """根据当前状态和动作返回下一个状态"""
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

# 测试状态转移
test_state = (0, 0)
print(f"测试状态转移: 从 {test_state} 开始")
for action in actions:
    next_state = get_next_state(test_state, action)
    print(f"  动作 {action}: 到 {next_state}")

# 3. 定义策略（随机策略）
print("\n定义随机策略...")

def random_policy(state):
    """随机策略：随机选择动作"""
    return np.random.choice(actions)

# 4. 可视化网格世界
print("\n可视化网格世界...")

plt.figure(figsize=(10, 10))
ax = plt.gca()

# 绘制网格
for i in range(grid_size):
    for j in range(grid_size):
        rect = Rectangle((j, grid_size - 1 - i), 1, 1, fill=False, color='black')
        ax.add_patch(rect)
        
        # 标注奖励
        reward = rewards[i, j]
        if reward != 0:
            color = 'green' if reward > 0 else 'red'
            plt.text(j + 0.5, grid_size - 1 - i + 0.5, 
                    f'{reward}', ha='center', va='center', 
                    fontsize=16, color=color)

# 设置坐标轴
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.xticks(np.arange(grid_size) + 0.5, range(grid_size))
plt.yticks(np.arange(grid_size) + 0.5, range(grid_size))
plt.title('Grid World - MDP Example')
plt.xlabel('Column')
plt.ylabel('Row')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mdp_grid_world.png'))
print("Grid world visualization saved to 'images/mdp_grid_world.png'")

# 5. 模拟一个episode
print("\n模拟一个episode...")

current_state = (0, 0)
total_reward = 0
trajectory = [current_state]
max_steps = 20

for step in range(max_steps):
    # 根据策略选择动作
    action = random_policy(current_state)
    
    # 获取下一个状态和奖励
    next_state = get_next_state(current_state, action)
    reward = rewards[next_state[0], next_state[1]]
    
    # 更新
    total_reward += reward
    trajectory.append(next_state)
    
    print(f"步骤 {step+1}: {current_state} -> {action} -> {next_state}, 奖励={reward}")
    
    current_state = next_state
    
    # 检查是否到达目标或陷阱
    if current_state == (3, 3):
        print("到达目标！")
        break
    elif current_state == (1, 1):
        print("掉入陷阱！")
        break

print(f"总奖励: {total_reward}")
print(f"轨迹长度: {len(trajectory)}")
print(f"轨迹: {trajectory}")

# 6. 可视化轨迹
print("\n可视化轨迹...")

plt.figure(figsize=(10, 10))
ax = plt.gca()

# 绘制网格
for i in range(grid_size):
    for j in range(grid_size):
        rect = Rectangle((j, grid_size - 1 - i), 1, 1, fill=False, color='black')
        ax.add_patch(rect)
        
        # 标注奖励
        reward = rewards[i, j]
        if reward != 0:
            color = 'green' if reward > 0 else 'red'
            plt.text(j + 0.5, grid_size - 1 - i + 0.5, 
                    f'{reward}', ha='center', va='center', 
                    fontsize=16, color=color)

# 绘制轨迹
trajectory_x = [j + 0.5 for (i, j) in trajectory]
trajectory_y = [grid_size - 1 - i + 0.5 for (i, j) in trajectory]
plt.plot(trajectory_x, trajectory_y, 'bo-', linewidth=2, markersize=8, label='轨迹')

# 设置坐标轴
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.xticks(np.arange(grid_size) + 0.5, range(grid_size))
plt.yticks(np.arange(grid_size) + 0.5, range(grid_size))
plt.title(f'网格世界 - 轨迹 (总奖励={total_reward})')
plt.xlabel('Column')
plt.ylabel('Row')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(images_dir, 'mdp_trajectory.png'))
print("Trajectory visualization saved to 'images/mdp_trajectory.png'")

# 7. MDP要素总结
print("\n" + "="*60)
print("MDP要素总结:")
print("="*60)
print("1. 状态空间 (S):")
print(f"   - 状态数量: {n_states}")
print(f"   - 示例: {states[:5]}...")
print()
print("2. 动作空间 (A):")
print(f"   - 动作数量: {n_actions}")
print(f"   - 动作: {actions}")
print()
print("3. 状态转移函数 (P):")
print("   - P(s'|s,a): 从状态s执行动作a到达s'的概率")
print("   - 本示例: 确定性转移")
print()
print("4. 奖励函数 (R):")
print("   - R(s,a,s'): 从状态s执行动作a到达s'获得的奖励")
print("   - 目标位置 (3,3): +10")
print("   - 陷阱位置 (1,1): -10")
print("   - 其他位置: 0")
print()
print("5. 策略 (π):")
print("   - π(a|s): 在状态s选择动作a的概率")
print("   - 本示例: 随机策略")
print("="*60)

print("\nMDP Demo完成！")
