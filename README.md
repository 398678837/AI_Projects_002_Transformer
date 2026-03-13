# AI学习路径：机器学习 → 深度学习 → Transformer → 大语言模型

## 项目概述
本项目提供从传统机器学习到深度学习，再到基于Transformer的大语言模型的结构化学习路径。

## 项目结构

```
AI_Projects_002_Transformer/
├── 001_machine_learning_机器学习/     # 001 - 传统机器学习
│   ├── 01_supervised_learning_监督学习/   # 01 - 监督学习
│   │   ├── 01_classification_分类/       # 01 - 分类任务
│   │   │   ├── 01_logistic_regression_逻辑回归/  # 逻辑回归
│   │   │   ├── 02_knn_近邻/             # KNN近邻
│   │   │   ├── 03_decision_tree_决策树/   # 决策树
│   │   │   ├── 04_random_forest_随机森林/  # 随机森林
│   │   │   ├── 05_xgboost_lightgbm_catboost/  # XGBoost/LightGBM/CatBoost
│   │   │   ├── 06_svm_支持向量机/        # SVM支持向量机
│   │   │   └── 07_naive_bayes_朴素贝叶斯/  # 朴素贝叶斯
│   │   └── 02_regression_回归/          # 02 - 回归任务
│   │       ├── 01_linear_regression_线性回归/  # 线性回归
│   │       ├── 02_ridge_lasso_正则回归/    # 岭回归/Lasso
│   │       └── 03_xgboost_lightgbm_回归/  # XGBoost/LightGBM回归
│   ├── 02_unsupervised_learning_无监督学习/ # 02 - 无监督学习
│   │   ├── 01_clustering_聚类/          # 01 - 聚类
│   │   │   ├── 01_kmeans_K均值/         # K-Means
│   │   │   ├── 02_dbscan_密度聚类/       # DBSCAN
│   │   │   └── 03_hierarchical_层次聚类/   # 层次聚类
│   │   └── 02_dimensionality_reduction_降维/  # 02 - 降维
│   │       ├── 01_pca_主成分分析/        # PCA
│   │       ├── 02_tsne_可视化/          # t-SNE
│   │       └── 03_lda_线性判别分析/       # LDA
│   └── 03_reinforcement_learning_强化学习/  # 03 - 强化学习
│       ├── 01_q_learning_Q学习/         # Q-Learning
│       ├── 02_dqn_深度Q网络/           # DQN
│       └── 03_ppo_近端策略优化/         # PPO
├── 002_deep_learning_深度学习/         # 002 - 深度学习
│   ├── 01_dnn_深度神经网络/           # 01 - 深度神经网络
│   │   ├── 01_basics_基础网络/         # 基础网络
│   │   ├── 02_activation_functions_激活函数/  # 激活函数
│   │   ├── 03_optimizers_优化器/        # 优化器
│   │   └── 04_regularization_正则化/     # 正则化
│   ├── 02_cnn_卷积神经网络/           # 02 - 卷积神经网络
│   │   ├── 01_convolution_卷积层/       # 卷积层
│   │   ├── 02_pooling_池化层/          # 池化层
│   │   ├── 03_classic_architectures_经典架构/  # 经典架构
│   │   │   ├── 01_lenet_LeNet/         # LeNet
│   │   │   ├── 02_alexnet_AlexNet/      # AlexNet
│   │   │   ├── 03_vgg_VGG/             # VGG
│   │   │   ├── 04_resnet_ResNet/        # ResNet
│   │   │   └── 05_mobilenet_MobileNet/   # MobileNet
│   │   └── 04_transfer_learning_迁移学习/   # 迁移学习
│   ├── 03_rnn_lstm_循环神经网络/       # 03 - 循环神经网络 & LSTM
│   │   ├── 01_rnn_循环神经网络/         # RNN
│   │   ├── 02_lstm_长短期记忆网络/       # LSTM
│   │   ├── 03_gru_门控循环单元/         # GRU
│   │   └── 04_sequence_models_序列模型/   # 序列模型
│   │       ├── 01_text_generation_文本生成/  # 文本生成
│   │       ├── 02_time_series_时间序列/     # 时间序列
│   │       └── 03_machine_translation_机器翻译/  # 机器翻译
│   └── 04_transformer_basics_Transformer基础/  # 04 - Transformer基础
│       ├── 01_transformer_vs_rnn_Transformer对比RNN/  # Transformer对比RNN
│       ├── 02_pre_training_预训练/       # 预训练
│       ├── 03_fine_tuning_微调/         # 微调
│       └── 04_llm_applications_LLM应用/   # LLM应用
├── 003_transformer_Transformer/       # 003 - Transformer & LLM
│   ├── 01_attention_mechanism_自注意力机制/  # 01 - 自注意力机制
│   │   ├── 01_self_attention_自注意力/    # 自注意力
│   │   ├── 02_multi_head_attention多头注意力/  # 多头注意力
│   │   ├── 03_positional_encoding位置编码/    # 位置编码
│   │   └── 04_attention_visualization注意力可视化/  # 注意力可视化
│   ├── 02_encoder_decoder_编码器解码器/    # 02 - 编码器-解码器架构
│   │   ├── 01_encoder_编码器/           # 编码器
│   │   ├── 02_decoder_解码器/           # 解码器
│   │   ├── 03_encoder_decoder_architecture_编码器解码器架构/  # 编码器-解码器架构
│   │   └── 04_cross_attention_交叉注意力/    # 交叉注意力
│   ├── 03_huggingface_HuggingFace/      # 03 - HuggingFace生态
│   │   ├── 01_transformers_library_Transformers库/  # Transformers库
│   │   ├── 02_model_hub_ModelHub/       # Model Hub
│   │   ├── 03_dataset_hub_DatasetHub/    # Dataset Hub
│   │   ├── 04_tokenization_分词器/       # 分词器
│   │   └── 05_model_loading_模型加载/    # 模型加载
│   └── 04_qwen_demo_Qwen对话/           # 04 - Qwen-1.8B-Chat对话Demo
│       ├── 01_environment_setup_环境配置/    # 环境配置
│       ├── 02_model_loading_模型加载/       # 模型加载
│       ├── 03_chat_implementation_对话实现/   # 对话实现
│       ├── 04_multi_turn_conversation_多轮对话/  # 多轮对话
│       └── 05_deployment_部署/           # 部署
├── config/                      # 配置文件
├── data/                        # 数据集存储
│   ├── raw/                     # 原始数据集
│   └── processed/               # 处理后的数据集
├── notebooks/                   # Jupyter笔记本
├── utils/                       # 工具函数
├── requirements.txt             # Python依赖包
└── README.md                    # 本文件
```

## 学习路径

### 第一天：机器学习/深度学习基础 + Transformer核心
- 15分钟：传统机器学习概念（监督学习/无监督学习/强化学习）
- 15分钟：深度学习演进（DNN → CNN → RNN → Transformer）
- 30分钟：Transformer架构（自注意力机制、编码器-解码器）

### 第二天：HuggingFace生态
- Transformers库
- Model Hub & Dataset Hub
- 加载预训练模型

### 第三天：实战Demo
- 构建Qwen-1.8B-Chat对话Demo
- 多轮对话功能
- GitHub上传 & 简历更新

## 环境安装

```bash
pip install -r requirements.txt
```

## 核心概念

### 技术演进路线
```
人工智能 (AI) → 机器学习 (ML) → 深度学习 (DL) → 大语言模型 (LLM/Transformer)
```

### ML vs DL vs LLM 对比
| 类型 | 特征处理 | 代表架构 | 学习重点 |
|------|---------|---------|---------|
| 传统机器学习 | 人工提取 | 决策树、逻辑回归 | 基础概念 |
| 深度学习 | 自动提取 | CNN、RNN、Transformer | 架构原理 |
| 大模型 LLM | 通用特征 | Transformer | 实际应用 |

## 简历技能
- 掌握大模型基础原理、Transformer架构核心逻辑
- 了解传统机器学习、深度学习基础概念与分类
- 熟悉HuggingFace生态，可使用Transformers库加载开源大模型实现对话功能

## 许可证
MIT License
