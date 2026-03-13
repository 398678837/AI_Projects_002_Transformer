# 决策树（Decision Tree）算法教材

## 课程目标
- 理解决策树的基本概念和原理
- 掌握决策树的构建过程和特征选择准则
- 学习如何使用scikit-learn实现决策树
- 掌握决策树的评估和调优方法
- 了解决策树的应用场景和局限性

## 1. 决策树基础

### 1.1 什么是决策树
决策树是一种基于树结构的监督学习算法，通过一系列的决策规则将数据集划分为不同的类别。它模拟了人类决策的过程，从根节点开始，根据特征的取值逐步向下判断，最终到达叶节点得到预测结果。

### 1.2 决策树的组成部分
- **根节点**：整个决策树的起点，包含所有训练样本
- **内部节点**：表示一个特征的测试，每个内部节点对应一个特征
- **分支**：表示特征的不同取值
- **叶节点**：表示最终的预测结果（分类或回归值）

### 1.3 决策树的优势
- **可解释性强**：树结构直观，决策过程透明
- **不需要特征标准化**：对特征尺度不敏感
- **能处理混合类型特征**：可以同时处理连续和离散特征
- **计算效率高**：训练和预测速度快
- **能捕捉非线性关系**：通过多级划分捕捉数据中的非线性模式

## 2. 决策树构建算法

### 2.1 ID3算法（Iterative Dichotomiser 3）
- **特征选择准则**：信息增益
- **适用场景**：离散特征的分类问题
- **局限性**：偏向于取值多的特征，不能处理连续特征

### 2.2 C4.5算法
- **特征选择准则**：信息增益比（解决ID3的偏向问题）
- **改进**：
  - 能处理连续特征（通过离散化）
  - 能处理缺失值
  - 支持剪枝
- **局限性**：计算复杂度较高

### 2.3 CART算法（Classification and Regression Tree）
- **特征选择准则**：Gini指数（分类）或均方误差（回归）
- **特点**：
  - 生成二叉树（每个节点只有两个分支）
  - 同时支持分类和回归任务
  - 效率高，是scikit-learn的默认实现

## 3. 特征选择准则

### 3.1 信息熵
信息熵是衡量数据集纯度的指标，熵值越大，数据越混乱：
 H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k 

### 3.2 信息增益
信息增益是父节点与子节点信息熵的差值，衡量特征对数据纯度的提升：
 IG(D, A) = H(D) - \sum_{v=1}^{V} \frac{|D^v|}{|D|} H(D^v) 

### 3.3 信息增益比
信息增益比是信息增益与特征固有值的比值，解决信息增益偏向于取值多的特征的问题：
 IG_Ratio(D, A) = \frac{IG(D, A)}{IV(A)} 
 其中，IV(A) = -\sum_{v=1}^{V} \frac{|D^v|}{|D|} \log_2 \frac{|D^v|}{|D|}

### 3.4 Gini指数
Gini指数衡量数据集的不纯度，取值范围[0, 1]，值越小纯度越高：
 Gini(D) = 1 - \sum_{k=1}^{K} p_k^2 

## 4. 决策树剪枝

### 4.1 预剪枝
在构建树的过程中提前停止，通过以下策略实现：
- **限制树的深度**（max_depth）
- **限制叶节点的最小样本数**（min_samples_leaf）
- **限制节点分裂的最小样本数**（min_samples_split）
- **限制特征的最小 impurity decrease**（min_impurity_decrease）

### 4.2 后剪枝
先构建完整的树，然后通过交叉验证剪去不必要的分支：
1. 从叶节点开始，自底向上评估每个非叶节点
2. 计算剪枝前后的验证集误差
3. 如果剪枝后误差减小或不变，则剪枝
4. 重复直到无法进一步剪枝

### 4.3 剪枝的作用
- **防止过拟合**：减少树的复杂度，提高泛化能力
- **简化模型**：使决策树更易于理解和解释
- **提高预测速度**：减少决策路径长度

## 5. 决策树的实现

### 5.1 scikit-learn中的决策树
```python
from sklearn.tree import DecisionTreeClassifier

# 创建决策树分类器
model = DecisionTreeClassifier(
    criterion='gini',      # 特征选择准则：'gini'或'entropy'
    max_depth=3,           # 树的最大深度
    min_samples_split=2,    # 节点分裂的最小样本数
    min_samples_leaf=1,     # 叶节点的最小样本数
    random_state=42         # 随机种子
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
```

### 5.2 决策树的可视化
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=feature_names,  # 特征名称
    class_names=class_names,      # 类别名称
    filled=True,                  # 填充颜色
    rounded=True,                 # 圆角矩形
    fontsize=12                   # 字体大小
)
plt.title('决策树可视化')
plt.show()
```

### 5.3 特征重要性评估
```python
# 获取特征重要性
feature_importance = model.feature_importances_

# 打印特征重要性
for feature, importance in zip(feature_names, feature_importance):
    print(f'{feature}: {importance:.4f}')
```

## 6. 决策树的评估

### 6.1 分类评估指标
- **准确率**：正确预测的样本数占总样本数的比例
- **精确率**：预测为正类的样本中实际为正类的比例
- **召回率**：实际为正类的样本中被正确预测的比例
- **F1值**：精确率和召回率的调和平均
- **混淆矩阵**：展示模型预测与真实值的对应关系

### 6.2 回归评估指标
- **均方误差（MSE）**：预测值与真实值之差的平方的平均值
- **均方根误差（RMSE）**：MSE的平方根
- **平均绝对误差（MAE）**：预测值与真实值之差的绝对值的平均值
- **R²评分**：模型解释数据方差的比例

### 6.3 交叉验证
使用K折交叉验证评估模型的泛化能力：
```python
from sklearn.model_selection import cross_val_score

# 5折交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'交叉验证准确率: {scores.mean():.4f} ± {scores.std():.4f}')
```

## 7. 决策树的参数调优

### 7.1 网格搜索
使用GridSearchCV搜索最佳参数组合：
```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# 创建网格搜索
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f'最佳参数: {grid_search.best_params_}')
print(f'最佳准确率: {grid_search.best_score_:.4f}')
```

### 7.2 重要参数说明
- **criterion**：特征选择准则，'gini'计算速度快，'entropy'更准确
- **max_depth**：树的最大深度，控制过拟合
- **min_samples_split**：节点分裂的最小样本数，值越大，树越简单
- **min_samples_leaf**：叶节点的最小样本数，值越大，树越简单
- **max_features**：分裂时考虑的最大特征数，减少过拟合

## 8. 决策树的应用场景

### 8.1 分类问题
- **信用评分**：评估贷款申请人的信用风险
- **医疗诊断**：根据症状和检查结果诊断疾病
- **客户流失预测**：预测客户是否会流失
- **垃圾邮件检测**：识别垃圾邮件

### 8.2 回归问题
- **房价预测**：根据房屋特征预测房价
- **销售额预测**：预测产品的销售额
- **股票价格预测**：预测股票的未来价格
- **需求预测**：预测产品的需求

### 8.3 特征工程
- **特征选择**：基于决策树的特征重要性选择关键特征
- **特征交互**：通过树结构捕捉特征间的交互关系
- **异常检测**：识别数据中的异常值

## 9. 决策树的局限性

### 9.1 过拟合风险
决策树容易过拟合，特别是当树深度较大时。解决方法：
- 限制树的深度
- 设置叶节点的最小样本数
- 进行剪枝
- 使用集成学习方法

### 9.2 不稳定性
决策树对训练数据的微小变化非常敏感，可能导致树结构的显著变化。解决方法：
- 使用集成学习（如随机森林）
- 增加训练数据量

### 9.3 偏向性
- **信息增益**：偏向于取值多的特征
- **CART**：在类别不平衡时可能偏向于多数类
解决方法：
- 使用信息增益比
- 对不平衡数据进行处理（如过采样、欠采样）

### 9.4 计算复杂度
- 训练时间：O(n * m * log(m))，其中n是特征数，m是样本数
- 预测时间：O(d)，其中d是树的深度
对于大规模数据集，可能需要：
- 特征选择
- 数据采样
- 使用更高效的实现

## 10. 集成学习与决策树

### 10.1 随机森林
随机森林是决策树的集成，通过以下方式提高性能：
- **自助采样**：对训练数据进行有放回采样
- **随机特征选择**：每个节点分裂时随机选择部分特征
- **多数投票**：多个决策树的预测结果通过投票决定

### 10.2 梯度提升树
梯度提升树通过迭代训练多个决策树，逐步减少误差：
- **加法模型**：每次添加一个新的决策树
- **梯度下降**：通过梯度下降优化损失函数
- **残差学习**：每个新树学习之前模型的残差

### 10.3 XGBoost和LightGBM
- **XGBoost**：极致梯度提升，优化了速度和性能
- **LightGBM**：基于梯度的单边抽样，速度更快，内存消耗更小

## 11. 实战案例：鸢尾花分类

### 11.1 数据加载与预处理
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 11.2 模型训练与评估
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 创建并训练模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'准确率: {accuracy:.4f}')
print('混淆矩阵:')
print(conf_matrix)
```

### 11.3 模型可视化
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title('鸢尾花分类决策树')
plt.show()
```

### 11.4 特征重要性分析
```python
feature_importance = model.feature_importances_

print('特征重要性:')
for feature, importance in zip(iris.feature_names, feature_importance):
    print(f'{feature}: {importance:.4f}')
```

## 12. 总结与展望

### 12.1 决策树的核心优势
- **直观易懂**：树结构清晰，决策过程透明
- **功能强大**：能处理分类和回归任务，适应各种数据类型
- **易于实现**：算法逻辑简单，实现门槛低
- **可解释性强**：能从树结构中提取规则，便于理解和应用

### 12.2 决策树的未来发展
- **自动化机器学习**：决策树作为AutoML的重要组成部分
- **可解释AI**：决策树在可解释AI中的应用
- **联邦学习**：决策树在隐私保护机器学习中的应用
- **图神经网络**：决策树与图神经网络的结合

### 12.3 学习建议
1. **掌握基础**：理解决策树的基本原理和算法流程
2. **实践应用**：在真实数据集上应用决策树算法
3. **参数调优**：学习如何选择最佳参数防止过拟合
4. **集成学习**：学习随机森林、梯度提升树等高级算法
5. **理论深入**：深入理解信息论和决策树的数学基础

## 13. 练习题

### 13.1 概念题
1. 决策树的基本组成部分有哪些？
2. 信息增益和Gini指数的区别是什么？
3. 预剪枝和后剪枝的区别是什么？
4. 决策树为什么容易过拟合？如何防止？
5. 随机森林是如何改进决策树的？

### 13.2 实践题
1. 使用scikit-learn的DecisionTreeClassifier对乳腺癌数据集进行分类
2. 尝试不同的max_depth值，观察对模型性能的影响
3. 使用GridSearchCV寻找最佳参数组合
4. 可视化决策树结构和特征重要性
5. 比较决策树与随机森林的性能差异

## 参考资料
- 《机器学习》周志华
- 《统计学习方法》李航
- 《Pattern Recognition and Machine Learning》Christopher M. Bishop
- scikit-learn官方文档
- 《数据挖掘导论》Pang-Ning Tan等
- 《Ensemble Methods: Foundations and Algorithms》 Zhi-Hua Zhou
