"""
策略迭代（Policy Iteration）演示
使用网格世界环境，完整展示策略迭代过程
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')
from matplotlib.patches import Rectangle

print("=" * 70)
print("策略迭代（Policy Iteration）演示")
print("=" * 70)

# 1. 定义网格世界环境
print("\n1. 定义网格世界环境...")

grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
state_index = {state: idx for idx, state in enumerate(states)}
n_states = len(states)

actions = ['up', 'down', 'left', 'right']
n_actions = len(actions)

# 奖励和转移
rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 10   # 目标
rewards[1, 1] = -5   # 陷阱

gamma = 0.9
theta = 0.001  # 收敛阈值

print(f"状态数量: {n_states}")
print(f"动作数量: {n_actions}")
print(f"折扣因子: {gamma}")

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

# 3. 策略评估
print("\n2. 策略评估函数...")

def policy_evaluation(policy, max_iter=1000):
    """评估给定策略的价值函数"""
    V = np.zeros(n_states)
    
    for iteration in range(max_iter):
        V_new = np.zeros(n_states)
        
        for state in states:
            idx = state_index[state]
            v = 0
            
            for a_idx, action in enumerate(actions):
                next_state = get_next_state(state, action)
                next_idx = state_index[next_state]
                r = rewards[next_state[0], next_state[1]]
                v += policy[a_idx] * (r + gamma * V[next_idx])
            
            V_new[idx] = v
        
        delta = np.max(np.abs(V_new - V))
        V = V_new
        
        if delta < theta:
            print(f"   策略评估在第 {iteration + 1} 次迭代收敛")
            break
    
    return V

# 4. 策略改进
print("\n3. 策略改进函数...")

def policy_improvement(V):
    """基于当前价值函数改进策略"""
    policy = np.zeros(n_actions)
    
    for state in states:
        q_values = []
        
        for action in actions:
            next_state = get_next_state(state, action)
            next_idx = state_index[next_state]
            r = rewards[next_state[0], next_state[1]]
            q = r + gamma * V[next_idx]
            q_values.append(q)
        
        best_action = np.argmax(q_values)
        policy[best_action] = 1.0
    
    return policy

# 5. 策略迭代主循环
print("\n4. 策略迭代主循环...")

def policy_iteration():
    """完整的策略迭代算法"""
    # 初始化随机策略
    policy = np.ones(n_actions) / n_actions
    V_history = []
    policy_history = []
    
    iteration = 0
    max_iterations = 100  # 限制最大迭代次数
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- 迭代 {iteration} ---")
        
        # 策略评估
        V = policy_evaluation(policy)
        V_history.append(V.copy())
        
        # 策略改进
        new_policy = policy_improvement(V)
        policy_history.append(new_policy.copy())
        
        # 检查策略是否收敛
        if np.allclose(policy, new_policy):
            print(f"   策略已收敛！")
            policy = new_policy
            V_history.append(V.copy())
            break
        
        policy = new_policy
    else:
        print(f"\n达到最大迭代次数 {max_iterations}，停止迭代")
    
    return V, policy, V_history, policy_history

# 运行策略迭代
V, optimal_policy, V_history, policy_history = policy_iteration()

# 6. 可视化
print("\n5. 可视化...")

action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
action_indices = {a: i for i, a in enumerate(actions)}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 选择几个关键迭代进行展示
show_iterations = [0, 1, 2, len(V_history) - 1]

for col, iteration in enumerate(show_iterations):
    if iteration >= len(V_history):
        continue
    
    # 绘制价值函数
    ax = axes[0, col] if col < 3 else axes[1, col - 3]
    V_grid = V_history[iteration].reshape(grid_size, grid_size)
    
    im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')
    
    for i in range(grid_size):
        for j in range(grid_size):
            state = (i, j)
            v = V_grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', 
                   color='white', fontsize=10)
    
    ax.set_title(f'迭代 {iteration + 1} - 价值函数', fontsize=12)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    
    # 绘制策略
    ax = axes[1, col] if col < 3 else None
    if ax is not None and iteration < len(policy_history):
        policy_grid = policy_history[iteration].reshape(grid_size, grid_size)
        
        im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')
        
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                best_action = np.argmax(policy_grid[i, j])
                arrow = action_arrows[actions[best_action]]
                ax.text(j, i, arrow, ha='center', va='center', 
                       color='white', fontsize=14, fontweight='bold')
        
        ax.set_title(f'迭代 {iteration + 1} - 策略', fontsize=12)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

# 最终结果
print("\n6. 最终结果...")
print("\n最优价值函数:")
V_grid = V.reshape(grid_size, grid_size)
print(V_grid)

print("\n最优策略:")
optimal_policy_grid = optimal_policy.reshape(grid_size, grid_size)
for i in range(grid_size):
    row = ""
    for j in range(grid_size):
        best_action = np.argmax(optimal_policy_grid[i, j])
        row += action_arrows[actions[best_action]] + " "
    print(row)

# 绘制最终结果
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

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
for i in range(grid_size):
    for j in range(grid_size):
        best_action = np.argmax(optimal_policy_grid[i, j])
        ax.text(j, i, action_arrows[actions[best_action]], ha='center', 
               va='center', color='white', fontsize=16, fontweight='bold')
ax.set_title('最优策略', fontsize=14)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/policy_iteration_result.png')
print("\n最终结果已保存为 'images/policy_iteration_result.png'")

# 7. 总结
print("\n" + "=" * 70)
print("策略迭代总结")
print("=" * 70)
print("""
策略迭代算法流程:
1. 初始化随机策略
2. 策略评估: 计算当前策略的价值函数
3. 策略改进: 基于价值函数更新策略
4. 重复步骤2-3直到策略收敛

特点:
- 策略评估和策略改进交替进行
- 每次改进后策略都会更好（或保持不变）
- 保证收敛到最优策略
- 对于小规模问题效率高
""")
print("=" * 70)
print("\nPolicy Iteration Demo完成！")
