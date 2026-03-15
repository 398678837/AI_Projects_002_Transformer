"""
价值迭代（Value Iteration）演示
使用网格世界环境，展示价值迭代过程
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("价值迭代（Value Iteration）演示")
print("=" * 70)

# 1. 定义环境
print("\n1. 定义网格世界环境...")

grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
state_index = {state: idx for idx, state in enumerate(states)}
n_states = len(states)

actions = ['up', 'down', 'left', 'right']
n_actions = len(actions)

rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 10   # 目标
rewards[1, 1] = -5   # 陷阱

gamma = 0.9
theta = 0.001

print(f"状态数量: {n_states}")
print(f"动作数量: {n_actions}")

# 2. 状态转移
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

# 3. 价值迭代
print("\n2. 价值迭代...")

def value_iteration():
    """价值迭代算法"""
    V = np.zeros(n_states)
    V_history = [V.copy()]
    
    iteration = 0
    
    while True:
        iteration += 1
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
        V = V_new.copy()
        V_history.append(V.copy())
        
        if delta < theta:
            print(f"   价值迭代在第 {iteration} 次迭代收敛")
            break
        
        if iteration % 10 == 0:
            print(f"   迭代 {iteration}, delta = {delta:.6f}")
    
    return V, V_history

V, V_history = value_iteration()

# 4. 提取策略
print("\n3. 提取最优策略...")

def extract_policy(V):
    """从价值函数提取策略"""
    policy = np.zeros(n_actions)
    
    for state in states:
        idx = state_index[state]
        q_values = []
        
        for action in actions:
            next_state = get_next_state(state, action)
            next_idx = state_index[next_state]
            r = rewards[next_state[0], next_state[1]]
            q = r + gamma * V[next_idx]
            q_values.append(q)
        
        best_action = np.argmax(q_values)
        policy[idx] = best_action
    
    return policy

optimal_policy = extract_policy(V)

# 5. 可视化
print("\n4. 可视化...")

action_arrows = ['↑', '↓', '←', '→']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

show_iters = [0, 1, 2, 5, 10, len(V_history) - 1]

for idx, iteration in enumerate(show_iters):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    if iteration >= len(V_history):
        iteration = len(V_history) - 1
    
    V_grid = V_history[iteration].reshape(grid_size, grid_size)
    
    im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')
    
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{V_grid[i,j]:.1f}", ha='center', va='center', 
                   color='white', fontsize=11)
    
    ax.set_title(f'迭代 {iteration} - 价值函数', fontsize=12)

plt.tight_layout()
plt.savefig('images/value_iteration_convergence.png')
print("收敛过程已保存为 'images/value_iteration_convergence.png'")

# 最终结果
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

V_grid = V.reshape(grid_size, grid_size)

# 价值函数
ax = axes2[0]
im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')
for i in range(grid_size):
    for j in range(grid_size):
        ax.text(j, i, f"{V_grid[i,j]:.2f}", ha='center', va='center', 
               color='white', fontsize=12)
ax.set_title('最优价值函数', fontsize=14)
plt.colorbar(im, ax=ax)

# 策略
ax = axes2[1]
im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')
policy_grid = optimal_policy.reshape(grid_size, grid_size)
for i in range(grid_size):
    for j in range(grid_size):
        best_action = int(policy_grid[i, j])
        ax.text(j, i, action_arrows[best_action], ha='center', va='center', 
               color='white', fontsize=16, fontweight='bold')
ax.set_title('最优策略', fontsize=14)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/value_iteration_result.png')
print("最终结果已保存为 'images/value_iteration_result.png'")

# 6. 总结
print("\n" + "=" * 70)
print("价值迭代总结")
print("=" * 70)
print("""
价值迭代算法流程:
1. 初始化价值函数 V(s) = 0
2. 迭代直到收敛:
   对每个状态 s:
     V(s) = max_a [R(s,a) + γ * V(s')]
3. 提取策略: π(s) = argmax_a Q(s,a)

特点:
- 直接求解贝尔曼最优方程
- 不需要显式的策略表示
- 收敛速度快于策略迭代
- 每次迭代更新所有状态
""")
print("=" * 70)
print("\nValue Iteration Demo完成！")
