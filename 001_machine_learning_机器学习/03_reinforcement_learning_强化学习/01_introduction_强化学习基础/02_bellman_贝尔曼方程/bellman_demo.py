"""
贝尔曼方程（Bellman Equation）的演示
使用简单的网格世界求解价值函数
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')
from matplotlib.patches import Rectangle

print("=" * 70)
print("贝尔曼方程演示")
print("=" * 70)

# 1. 定义网格世界环境
print("\n1. 定义网格世界环境...")

grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
state_index = {state: idx for idx, state in enumerate(states)}
n_states = len(states)

actions = ['up', 'down', 'left', 'right']
n_actions = len(actions)

# 奖励函数
rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 10   # 目标位置
rewards[1, 1] = -5    # 陷阱位置
rewards[2, 2] = -2    # 小惩罚位置

gamma = 0.9  # 折扣因子

print(f"网格大小: {grid_size}x{grid_size}")
print(f"状态数量: {n_states}")
print(f"动作数量: {n_actions}")
print(f"折扣因子 gamma: {gamma}")
print("奖励矩阵:")
print(rewards)

# 2. 状态转移函数
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

# 3. 定义随机策略
def random_policy():
    """随机策略：每个动作的概率相等"""
    return np.ones(n_actions) / n_actions

# 4. 求解贝尔曼期望方程（迭代法）
print("\n2. 求解贝尔曼期望方程...")

def solve_bellman_expectation(policy, max_iter=1000, tol=1e-6):
    """迭代求解贝尔曼期望方程"""
    V = np.zeros(n_states)
    
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        
        for state in states:
            idx = state_index[state]
            action_probs = policy()
            v = 0
            
            for a_idx, action in enumerate(actions):
                next_state = get_next_state(state, action)
                next_idx = state_index[next_state]
                r = rewards[next_state[0], next_state[1]]
                v += action_probs[a_idx] * (r + gamma * V[next_idx])
            
            V_new[idx] = v
        
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < tol:
            print(f"  贝尔曼期望方程在第 {iteration + 1} 次迭代收敛")
            break
    
    return V

# 求解
V_random = solve_bellman_expectation(random_policy)

# 5. 求解贝尔曼最优方程（值迭代）
print("\n3. 求解贝尔曼最优方程...")

def value_iteration(max_iter=1000, tol=1e-6):
    """值迭代求解贝尔曼最优方程"""
    V = np.zeros(n_states)
    
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        
        for state in states:
            idx = state_index[state]
            q_values = []
            
            for action in actions:
                next_state = get_next_state(state, action)
                next_idx = state_index[next_state]
                r = rewards[next_state[0], next_state[1]]
                q = r + gamma * V[next_idx]
                q_values.append(q)
            
            V_new[idx] = max(q_values)
        
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < tol:
            print(f"  值迭代在第 {iteration + 1} 次迭代收敛")
            break
    
    return V

# 求解
V_optimal = value_iteration()

# 6. 从最优价值函数提取策略
print("\n4. 提取最优策略...")

def extract_policy(V):
    """从价值函数提取最优策略"""
    policy = []
    
    for state in states:
        q_values = []
        
        for action in actions:
            next_state = get_next_state(state, action)
            next_idx = state_index[next_state]
            r = rewards[next_state[0], next_state[1]]
            q = r + gamma * V[state_index[next_state]]
            q_values.append(q)
        
        best_action_idx = np.argmax(q_values)
        policy.append(actions[best_action_idx])
    
    return policy

optimal_policy = extract_policy(V_optimal)

# 7. 可视化结果
print("\n5. 可视化结果...")

# 将价值函数重塑为网格形状
V_random_grid = V_random.reshape(grid_size, grid_size)
V_optimal_grid = V_optimal.reshape(grid_size, grid_size)
policy_grid = np.array(optimal_policy).reshape(grid_size, grid_size)

# 箭头符号
action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制奖励
ax = axes[0]
im = ax.imshow(rewards, cmap='RdYlGn', interpolation='nearest')
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, f"{rewards[i,j]:.0f}", ha='center', va='center', 
                color='black', fontsize=14, fontweight='bold')
ax.set_title('奖励函数', fontsize=14)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))

# 绘制随机策略价值函数
ax = axes[1]
im = ax.imshow(V_random_grid, cmap='viridis', interpolation='nearest')
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, f"{V_random_grid[i,j]:.2f}", ha='center', va='center', 
                color='white', fontsize=12)
ax.set_title('随机策略价值函数', fontsize=14)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))

# 绘制最优策略和价值函数
ax = axes[2]
im = ax.imshow(V_optimal_grid, cmap='viridis', interpolation='nearest')
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, f"{V_optimal_grid[i,j]:.2f}\n{action_arrows[policy_grid[i,j]]}", 
                ha='center', va='center', color='white', fontsize=10)
ax.set_title('最优策略 & 价值函数', fontsize=14)
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))

plt.tight_layout()
plt.savefig('images/bellman_equations.png')
print("可视化结果已保存为 'images/bellman_equations.png'")

# 8. 总结
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("\n1. 贝尔曼期望方程:")
print("   V^π(s) = Σ π(a|s) * Σ P(s'|s,a) * [R(s,a,s') + γ V^π(s')]")
print("\n2. 贝尔曼最优方程:")
print("   V*(s) = max_a Σ P(s'|s,a) * [R(s,a,s') + γ V*(s')]")
print("\n3. 主要区别:")
print("   - 期望方程：给定策略，计算价值函数")
print("   - 最优方程：寻找最优策略，最大化价值函数")
print("\n4. 求解方法:")
print("   - 策略评估：求解贝尔曼期望方程")
print("   - 值迭代：求解贝尔曼最优方程")
print("=" * 70)

print("\nBellman Equations Demo完成！")
