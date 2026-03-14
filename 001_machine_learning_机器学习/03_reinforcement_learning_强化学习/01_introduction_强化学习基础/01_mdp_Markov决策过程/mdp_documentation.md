# Markov决策过程（MDP）详细文档

## 1. 概念介绍

### 1.1 什么是Markov决策过程
Markov决策过程（Markov Decision Process，MDP）是强化学习的数学框架，用于描述智能体在环境中的决策过程。它提供了一种形式化的方法来建模序贯决策问题。

### 1.2 MDP的五个要素

一个MDP由以下五个要素组成：

1. **状态空间（S）**：环境中所有可能的状态集合
2. **动作空间（A）**：智能体可以执行的所有动作集合
3. **状态转移函数（P）**：$P(s'|s,a)$，在状态$s$执行动作$a$到达状态$s'$的概率
4. **奖励函数（R）**：$R(s,a,s')$，在状态$s$执行动作$a$到达状态$s'$获得的奖励
5. **策略（π）**：$\pi(a|s)$，在状态$s$选择动作$a$的概率

### 1.3 Markov性质
MDP的核心是Markov性质：
- **下一个状态只依赖于当前状态和动作**
- **不依赖于更早的历史状态**
- 数学表示：$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$

### 1.4 应用场景
- **游戏AI**：下棋、电子游戏
- **机器人控制**：机器人导航、操作
- **推荐系统**：序列推荐
- **资源管理**：能源管理、库存控制
- **医疗决策**：治疗方案选择

## 2. 技术原理

### 2.1 状态价值函数

状态价值函数$V^\pi(s)$表示在状态$s$下遵循策略$\pi$的预期回报：

$$ V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] $$

其中$G_t$是回报（Return）：

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... $$

$\gamma \in [0,1]$是折扣因子，决定了未来奖励的重要性。

### 2.2 动作价值函数

动作价值函数$Q^\pi(s,a)$表示在状态$s$下执行动作$a$，然后遵循策略$\pi$的预期回报：

$$ Q^\pi(s,a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a] $$

### 2.3 Bellman期望方程

状态价值函数的Bellman期望方程：

$$ V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')] $$

动作价值函数的Bellman期望方程：

$$ Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')] $$

## 3. 代码实现

文件：`mdp_demo.py`

### 3.1 核心步骤
1. **定义状态空间**：网格世界的所有状态
2. **定义动作空间**：上、下、左、右
3. **定义状态转移**：确定性或随机转移
4. **定义奖励函数**：目标奖励、陷阱惩罚
5. **定义策略**：随机策略或其他策略
6. **模拟episode**：智能体与环境交互
7. **可视化**：网格世界和轨迹

### 3.2 关键代码

```python
# 定义状态空间
grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]

# 定义动作空间
actions = ['up', 'down', 'left', 'right']

# 定义状态转移函数
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

# 定义奖励
rewards = np.zeros((grid_size, grid_size))
rewards[3, 3] = 10  # 目标
rewards[1, 1] = -10  # 陷阱

# 随机策略
def random_policy(state):
    return np.random.choice(actions)
```

## 4. MDP的分类

### 4.1 按转移概率分类

- **确定性MDP**：状态转移是确定的，$P(s'|s,a) = 1$
- **随机性MDP**：状态转移是随机的，$P(s'|s,a)$是概率分布

### 4.2 按时间分类

- **有限 horizon MDP**：有固定的时间步数
- **无限 horizon MDP**：没有时间限制，持续交互

## 5. 总结

MDP是强化学习的数学基础，它提供了一种形式化的方法来描述序贯决策问题。理解MDP对于学习强化学习算法至关重要。

---

**参考资料**：
- 《Reinforcement Learning: An Introduction》Sutton & Barto
- 《Markov Decision Processes: Discrete Stochastic Dynamic Programming》Puterman
