# AI_Projects_002_Transformer

## 项目概述

这是一个全面的机器学习、深度学习和Transformer学习项目，包含了从基础算法到高级模型的完整实现和文档。项目采用中英双语命名，便于不同背景的学习者理解和使用。

## 项目结构

```
AI_Projects_002_Transformer/
├── 001_machine_learning_机器学习/             # 机器学习部分
│   ├── 01_supervised_learning_监督学习/      # 监督学习
│   │   ├── 01_classification_分类/           # 分类算法
│   │   └── 02_regression_回归/              # 回归算法
│   ├── 02_unsupervised_learning_无监督学习/    # 无监督学习
│   │   ├── 01_clustering_聚类/               # 聚类算法
│   │   └── 02_dimensionality_reduction_降维/  # 降维算法
│   └── 03_reinforcement_learning_强化学习/    # 强化学习
├── 002_deep_learning_深度学习/               # 深度学习部分
│   ├── 01_dnn_深度神经网络/                 # 深度神经网络
│   ├── 02_cnn_卷积神经网络/                 # 卷积神经网络
│   ├── 03_rnn_lstm_循环神经网络/             # 循环神经网络
│   └── 04_transformer_basics_Transformer基础/  # Transformer基础
├── 003_large_language_model_大语言模型/       # 大语言模型部分
│   ├── 01_attention_mechanism_自注意力机制/    # 自注意力机制
│   ├── 02_encoder_decoder_编码器解码器/        # 编码器解码器
│   ├── 03_huggingface_HuggingFace/           # HuggingFace生态
│   └── 04_qwen_demo_Qwen对话/                # Qwen对话
├── 004_numpy_pandas_数据处理基础/             # NumPy和Pandas
│   ├── 01_numpy_basics_NumPy基础/            # NumPy基础
│   ├── 02_numpy_advanced_NumPy进阶/          # NumPy进阶
│   ├── 03_pandas_basics_Pandas基础/            # Pandas基础
│   ├── 04_pandas_advanced_Pandas进阶/          # Pandas进阶
│   └── 05_data_visualization_数据可视化/       # 数据可视化
├── 005_pytorch_PyTorch学习/                   # PyTorch深度学习
│   ├── 01_pytorch_basics_PyTorch基础/         # PyTorch基础
│   ├── 02_pytorch_advanced_PyTorch进阶/       # PyTorch进阶
│   ├── 03_pytorch_models_PyTorch模型/        # PyTorch模型
│   └── 04_pytorch_gpu_PyTorch_GPU/          # PyTorch GPU
├── 006_tensorflow_TensorFlow学习/               # TensorFlow深度学习
│   ├── 01_tensorflow_basics_TensorFlow基础/     # TensorFlow基础
│   ├── 02_tensorflow_advanced_TensorFlow进阶/   # TensorFlow进阶
│   └── 03_tensorflow_models_TensorFlow模型/    # TensorFlow模型
├── 007_hadoop_spark_大数据处理/               # Hadoop和Spark大数据
│   ├── 01_hadoop_basics_Hadoop基础/            # Hadoop基础
│   ├── 02_hadoop_ecosystem_Hadoop生态/          # Hadoop生态
│   ├── 03_spark_basics_Spark基础/               # Spark基础
│   └── 04_spark_advanced_Spark进阶/             # Spark进阶
├── requirements.txt                           # 依赖包
└── README.md                                  # 项目说明
```

## 7天学习路径

### Day 1-2: 数据处理基础
- **NumPy基础**：数组创建、操作、数学运算、广播机制
- **NumPy进阶**：高级索引、形状操作、线性代数、随机数
- **Pandas基础**：Series、DataFrame、数据选择、数据操作
- **Pandas进阶**：分组操作、合并连接、缺失数据、时间序列
- **数据可视化**：Matplotlib、Seaborn、Pandas绘图

### Day 3-4: 机器学习与深度学习
- **机器学习**：监督学习、无监督学习、强化学习
- **深度学习**：DNN、CNN、RNN/LSTM、Transformer基础

### Day 5: 大语言模型
- **自注意力机制**：自注意力、多头注意力、位置编码、可视化
- **编码器-解码器**：编码器、解码器、架构、交叉注意力
- **HuggingFace生态**：Transformers库、Model Hub、Dataset Hub、分词器、模型加载
- **Qwen对话**：环境配置、模型加载、对话实现、多轮对话、部署

### Day 6: PyTorch与TensorFlow
- **PyTorch**：基础、进阶、模型、GPU
- **TensorFlow**：基础、进阶、模型

### Day 7: 大数据处理
- **Hadoop**：基础、生态（Hive、HBase、Flume、Kafka）
- **Spark**：基础、进阶（流处理、MLlib、GraphX、结构化流）

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

1. **全面覆盖**：包含机器学习、深度学习、大语言模型、NumPy/Pandas、PyTorch、TensorFlow、Hadoop/Spark
2. **详细文档**：每个算法都有详细的markdown文档，便于学习和理解
3. **代码实现**：提供简洁易懂的代码实现
4. **可视化**：包含丰富的可视化结果，便于直观理解算法效果
5. **中英双语**：采用中英双语命名，适合不同背景的学习者使用
6. **环境配置**：提供完整的依赖包和环境配置说明

## 后续计划

1. **完善内容**：为新创建的模块添加详细的demo、文档和教材
2. **实战项目**：添加更多实战项目
3. **性能优化**：添加模型性能优化相关内容

## 贡献

欢迎提交issue和pull request，一起完善这个项目！

## 许可证

本项目采用MIT许可证。
