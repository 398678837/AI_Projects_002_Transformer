# AI_Projects_002_Transformer

## 项目概述

这是一个全面的机器学习、深度学习和Transformer学习项目，包含了从基础算法到高级模型的完整实现和文档。项目采用中英双语命名，便于不同背景的学习者理解和使用。

## 项目结构

```
AI_Projects_002_Transformer/
├── 001_machine_learning_机器学习/             # 机器学习部分
│   ├── 01_supervised_learning_监督学习/      # 监督学习
│   │   ├── 01_classification_分类/           # 分类算法
│   │   │   ├── 01_logistic_regression_逻辑回归/
│   │   │   ├── 02_knn_K近邻/
│   │   │   ├── 03_decision_tree_决策树/
│   │   │   ├── 04_random_forest_随机森林/
│   │   │   ├── 05_xgboost_lightgbm_catboost_XGBoost_LightGBM_CatBoost/
│   │   │   ├── 06_svm_支持向量机/
│   │   │   └── 07_naive_bayes_朴素贝叶斯/
│   │   └── 02_regression_回归/              # 回归算法
│   │       ├── 01_linear_regression_线性回归/
│   │       ├── 02_polynomial_regression_多项式回归/
│   │       ├── 03_regularized_regression_正则化回归/
│   │       ├── 04_support_vector_regression_支持向量回归/
│   │       ├── 05_tree_regression_树模型回归/
│   │       ├── 06_xgboost_lightgbm_catboost_regression_XGBoost_LightGBM_CatBoost回归/
│   │       ├── 07_neural_network_regression_神经网络回归/
│   │       └── 08_regression_comparison_回归模型综合对比/
│   ├── 02_unsupervised_learning_无监督学习/    # 无监督学习
│   │   ├── 01_clustering_聚类/               # 聚类算法
│   │   │   ├── 01_kmeans_K均值/
│   │   │   └── 02_advanced_clustering_高级聚类/
│   │   └── 02_dimensionality_reduction_降维/  # 降维算法
│   │       ├── 01_pca_主成分分析/
│   │       ├── 02_tsne_t分布邻域嵌入/
│   │       ├── 03_lda_线性判别分析/
│   │       ├── 04_mds_多维缩放/
│   │       └── 05_advanced_dimensionality_reduction_高级降维/
│   └── 03_reinforcement_learning_强化学习/    # 强化学习
│       ├── 01_introduction_强化学习基础/      # 基础概念
│       │   ├── 01_mdp_Markov决策过程/
│       │   └── 02_bellman_贝尔曼方程/
│       ├── 02_dynamic_programming_动态规划/  # 基于模型的方法
│       │   ├── 01_policy_iteration_策略迭代/
│       │   └── 02_value_iteration_价值迭代/
│       ├── 03_monte_carlo_蒙特卡洛方法/      # 无模型方法
│       │   ├── 01_mc_basics_蒙特卡洛基础/
│       │   └── 02_mc_control_蒙特卡洛控制/
│       ├── 04_temporal_difference_时序差分/    # TD学习
│       │   ├── 01_q_learning_Q学习/
│       │   └── 02_sarsa_SARSA/
│       ├── 05_deep_rl_深度强化学习/          # 深度RL
│       │   ├── 01_dqn_深度Q网络/
│       │   ├── 02_policy_gradient_策略梯度/
│       │   ├── 03_actor_critic_Actor-Critic/
│       │   └── 04_ppo_PPO算法/
│       └── 06_rl_comparison_强化学习综合/     # 综合
│           ├── 01_gym_environments_Gym环境/
│           ├── 02_algorithm_comparison_算法对比/
│           └── 03_applications_实际应用/
├── 002_deep_learning_深度学习/               # 深度学习部分
│   ├── 01_fundamentals_基础/
│   ├── 02_cnn_卷积神经网络/
│   ├── 03_rnn_lstm_gru_循环神经网络/
│   └── 04_transformer_基础/
├── 003_large_language_model_大语言模型/       # 大语言模型部分
│   ├── 01_huggingface_生态/
│   └── 02_qwen_1_8b_chat_部署与微调/
├── requirements.txt                           # 依赖包
└── README.md                                  # 项目说明
```

## 3天学习路径

### Day 1: 机器学习基础与深度学习回顾
- **上午**：机器学习基础概念、监督学习vs无监督学习、模型评估
- **下午**：线性回归、逻辑回归、决策树、随机森林
- **晚上**：深度学习基础回顾、CNN/RNN/Transformer原理

### Day 2: HuggingFace生态与模型应用
- **上午**：HuggingFace Transformers库、预训练模型使用
- **下午**：文本分类、情感分析、命名实体识别
- **晚上**：模型微调基础、参数高效微调(PEFT)

### Day 3: Qwen-1.8B-Chat部署与应用
- **上午**：模型下载与部署、环境配置
- **下午**：对话系统开发、API调用
- **晚上**：实际应用案例、性能优化

## 已实现的算法

### 监督学习

#### 分类算法
1. **逻辑回归**：二分类和多分类实现，包含详细文档和示例
2. **K近邻(KNN)**：基于距离的分类算法，包含参数调优
3. **决策树**：基于信息增益的分类算法，包含可视化
4. **随机森林**：集成学习算法，提高分类性能
5. **XGBoost/LightGBM/CatBoost**：梯度提升树算法，工业级应用
6. **支持向量机(SVM)**：基于最大化间隔的分类算法
7. **朴素贝叶斯**：基于贝叶斯定理的概率分类算法

#### 回归算法
1. **线性回归**：基础线性模型，包含数学原理和实现
2. **多项式回归**：处理非线性关系
3. **正则化回归**：岭回归、LASSO、弹性网络
4. **支持向量回归(SVR)**：基于核函数的回归
5. **树回归**：决策树回归、随机森林回归
6. **集成树回归**：XGBoost、LightGBM、CatBoost回归
7. **神经网络回归**：多层感知机回归
8. **回归模型综合对比**：13种回归模型对比

### 无监督学习

#### 聚类算法
1. **K-Means**：基于距离的聚类算法，包含肘部法则和轮廓系数分析
2. **层次聚类**：基于层次结构的聚类算法，包含树状图可视化
3. **DBSCAN**：基于密度的聚类算法，可发现任意形状的聚类
4. **高斯混合模型**：基于概率模型的聚类算法，支持软聚类

#### 降维算法
1. **主成分分析(PCA)**：线性降维算法，包含解释方差分析和可视化
2. **t-SNE**：非线性降维，擅长可视化高维数据
3. **线性判别分析(LDA)**：监督降维，最大化类间距离
4. **多维缩放(MDS)**：保持距离结构的降维
5. **高级降维**：Isomap、LLE等非线性方法

### 强化学习

#### 基础概念
1. **MDP (Markov决策过程)**：强化学习的数学框架
2. **贝尔曼方程**：价值函数的递归关系

#### 动态规划（基于模型）
1. **策略迭代**：交替进行策略评估和策略改进
2. **价值迭代**：直接求解贝尔曼最优方程

#### 蒙特卡洛方法（无模型）
1. **蒙特卡洛基础**：基于完整episode学习
2. **蒙特卡洛控制**：结合策略改进

#### 时序差分学习
1. **Q学习**：Off-Policy TD控制
2. **SARSA**：On-Policy TD控制

#### 深度强化学习
1. **DQN**：深度Q网络，结合经验回放和目标网络
2. **策略梯度**：直接优化策略参数
3. **Actor-Critic**：结合价值函数和策略梯度
4. **PPO**：近端策略优化，当今最流行的RL算法

## 环境配置

### 虚拟环境创建（推荐）
```bash
# 在D盘创建虚拟环境
python -m venv D:\Programming\Trae\Environment\ml_env

# 激活虚拟环境
D:\Programming\Trae\Environment\ml_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 依赖包
```
# 基础包
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0

# 高级包
xgboost==1.7.6
lightgbm==3.3.5
catboost==1.1.1
scipy==1.10.1

# 深度学习包
tensorflow==2.13.0
keras==2.13.1
pytorch==2.0.1
transformers==4.30.2
datasets==2.13.1
notebook==6.5.4
```

## 使用方法

1. **克隆仓库**：
   ```bash
   git clone https://github.com/398678837/AI_Projects_002_Transformer.git       
   cd AI_Projects_002_Transformer
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **运行示例**：
   ```bash
   # 运行逻辑回归示例
   python 001_machine_learning_机器学习/01_supervised_learning_监督学习/01_classification_分类/01_logistic_regression_逻辑回归/logistic_regression_demo.py

   # 运行多项式回归示例
   python 001_machine_learning_机器学习/01_supervised_learning_监督学习/02_regression_回归/02_polynomial_regression_多项式回归/polynomial_regression_demo.py

   # 运行K-Means聚类示例
   python 001_machine_learning_机器学习/02_unsupervised_learning_无监督学习/01_clustering_聚类/01_kmeans_K均值/kmeans_demo.py

   # 运行PCA降维示例
   python 001_machine_learning_机器学习/02_unsupervised_learning_无监督学习/02_dimensionality_reduction_降维/01_pca_主成分分析/pca_demo.py

   # 运行强化学习示例
   python 001_machine_learning_机器学习/03_reinforcement_learning_强化学习/01_introduction_强化学习基础/01_mdp_Markov决策过程/mdp_demo.py
   
   # 运行Q学习示例
   python 001_machine_learning_机器学习/03_reinforcement_learning_强化学习/04_temporal_difference_时序差分/01_q_learning_Q学习/q_learning_demo.py
   
   # 运行PPO示例
   python 001_machine_learning_机器学习/03_reinforcement_learning_强化学习/05_deep_rl_深度强化学习/04_ppo_PPO算法/ppo_demo.py
   ```

4. **查看文档**：
   每个算法目录下都有详细的markdown文档，包括概念介绍、技术原理、代码实现等内容。

## 项目特点

1. **全面覆盖**：包含机器学习、深度学习和Transformer的核心算法和模型
2. **详细文档**：每个算法都有详细的markdown文档，便于学习和理解
3. **代码实现**：提供scikit-learn实现，代码简单易懂
4. **可视化**：包含丰富的可视化结果，便于直观理解算法效果
5. **中英双语**：采用中英双语命名，适合不同背景的学习者使用
6. **环境配置**：提供完整的依赖包和环境配置说明

## 后续计划

1. **深度学习部分**：完成CNN、RNN/LSTM/GRU、Transformer的详细实例和文档
2. **大语言模型部分**：完成HuggingFace生态和Qwen-1.8B-Chat的部署与应用
3. **模型部署**：添加模型部署和服务化相关内容
4. **实战项目**：添加更多实战项目，如图像分类、文本分类、推荐系统等   
5. **性能优化**：添加模型性能优化相关内容

## 贡献

欢迎提交issue和pull request，一起完善这个项目！

## 许可证

本项目采用MIT许可证。
