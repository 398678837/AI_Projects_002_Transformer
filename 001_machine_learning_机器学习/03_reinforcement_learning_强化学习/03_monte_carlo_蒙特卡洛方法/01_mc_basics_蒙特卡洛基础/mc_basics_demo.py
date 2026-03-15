"""
蒙特卡洛方法（Monte Carlo Methods）基础演示
使用21点游戏环境展示蒙特卡洛方法的基本概念
"""

import numpy as np
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("蒙特卡洛方法基础演示")
print("=" * 70)

# 1. 简化21点环境
print("\n1. 定义简化21点环境...")

def get_card():
    """随机获取一张牌"""
    card = np.random.randint(1, 11)
    return card

def get_initial_state():
    """获取初始状态"""
    player_sum = get_card() + get_card()
    dealer_showing = get_card()
    usable_ace = 1 if player_sum <= 11 else 0
    return (player_sum, dealer_showing, usable_ace)

def get_action(state, policy=None):
    """获取动作: 0=停牌, 1=要牌"""
    player_sum, dealer_showing, usable_ace = state
    if policy is None:
        policy = simple_policy
    return policy(state)

def simple_policy(state):
    """简单策略: 总和小于20继续要牌"""
    player_sum, dealer_showing, usable_ace = state
    return 0 if player_sum >= 20 else 1

def step(state, action):
    """执行一步"""
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
        # 庄家回合
        dealer_sum = dealer_showing + get_card() + get_card()
        usable_ace_dealer = 1 if dealer_sum <= 11 else 0
        
        while dealer_sum < 17:
            new_card = get_card()
            dealer_sum += new_card
            if new_card == 1 and dealer_sum + 10 <= 21:
                dealer_sum += 10
                usable_ace_dealer = 1
        
        if dealer_sum > 21 or player_sum > dealer_sum:
            reward = 1
        elif player_sum < dealer_sum:
            reward = -1
        else:
            reward = 0
        
        return state, reward, True

# 2. 生成Episode
print("\n2. 生成Episode...")

def generate_episode(policy=None):
    """生成一个完整的episode"""
    state = get_initial_state()
    trajectory = []
    
    while True:
        action = get_action(state, policy)
        trajectory.append((state, action))
        next_state, reward, done = step(state, action)
        
        if done:
            trajectory.append((next_state, None, reward))
            break
        
        state = next_state
    
    return trajectory

# 测试生成一个episode
episode = generate_episode(simple_policy)
print(f"Episode长度: {len(episode)}")
print("轨迹示例 (前5步):")
for i, (state, action, reward) in enumerate(episode[:5]):
    print(f"  步骤{i}: 状态={state}, 动作={'停牌' if action==0 else '要牌'}")

# 3. 蒙特卡洛策略评估
print("\n3. 蒙特卡洛策略评估...")

def mc_policy_evaluation(policy, n_episodes=10000):
    """蒙特卡洛策略评估 - 首次访问方法"""
    V = {}
    returns_count = {}
    
    for episode_idx in range(n_episodes):
        episode = generate_episode(policy)
        states_visited = set()
        
        for i, (state, action, reward) in enumerate(episode[:-1]):
            if state not in states_visited:
                states_visited.add(state)
                
                # 计算回报
                G = sum(r for (_, _, r) in episode[i:])
                
                if state not in V:
                    V[state] = 0
                    returns_count[state] = 0
                
                returns_count[state] += 1
                V[state] += (G - V[state]) / returns_count[state]
    
    return V

V_estimated = mc_policy_evaluation(simple_policy, n_episodes=50000)

print(f"评估了 {len(V_estimated)} 个状态")
print("\n部分状态价值函数:")
for i, (state, value) in enumerate(sorted(V_estimated.items())[:5]):
    print(f"  玩家={state[0]}, 庄家显示={state[1]}, 有可用Ace={state[2]}: V={value:.3f}")

# 4. 可视化
print("\n4. 可视化...")

# 提取可用的价值函数
player_sums = range(12, 22)
dealer_showing = range(1, 11)

V_matrix = np.zeros((len(player_sums), len(dealer_showing)))
for i, ps in enumerate(player_sums):
    for j, ds in enumerate(dealer_showing):
        state = (ps, ds, 0)
        if state in V_estimated:
            V_matrix[i, j] = V_estimated[state]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 有可用Ace
ax = axes[0]
V_matrix_ace = np.zeros((len(player_sums), len(dealer_showing)))
for i, ps in enumerate(player_sums):
    for j, ds in enumerate(dealer_showing):
        state = (ps, ds, 1)
        if state in V_estimated:
            V_matrix_ace[i, j] = V_estimated[state]

im = ax.imshow(V_matrix_ace, cmap='RdYlGn', origin='lower', aspect='auto')
ax.set_xlabel('庄家显示')
ax.set_ylabel('玩家总点数')
ax.set_title('有可用Ace时的价值函数')
ax.set_xticks(range(len(dealer_showing)))
ax.set_xticklabels(dealer_showing)
ax.set_yticks(range(len(player_sums)))
ax.set_yticklabels(player_sums)
plt.colorbar(im, ax=ax)

# 无可用Ace
ax = axes[1]
im = ax.imshow(V_matrix, cmap='RdYlGn', origin='lower', aspect='auto')
ax.set_xlabel('庄家显示')
ax.set_ylabel('玩家总点数')
ax.set_title('无可用Ace时的价值函数')
ax.set_xticks(range(len(dealer_showing)))
ax.set_xticklabels(dealer_showing)
ax.set_yticks(range(len(player_sums)))
ax.set_yticklabels(player_sums)
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('images/mc_basic_evaluation.png')
print("可视化已保存为 'images/mc_basic_evaluation.png'")

# 5. 总结
print("\n" + "=" * 70)
print("蒙特卡洛方法基础总结")
print("=" * 70)
print("""
蒙特卡洛方法特点:
1. 无模型方法: 不需要知道环境转移概率
2. 基于Episode: 通过完整的episode学习
3. 样本平均: 用回报的样本均值估计价值函数
4. 首次访问: 每个episode只对首次访问的状态进行更新

核心思想:
- V(s) ≈ (1/N) * Σ G_i(s)
  其中G_i(s)是第i次访问状态s的回报
""")
print("=" * 70)
print("\nMonte Carlo Basic Demo完成！")
