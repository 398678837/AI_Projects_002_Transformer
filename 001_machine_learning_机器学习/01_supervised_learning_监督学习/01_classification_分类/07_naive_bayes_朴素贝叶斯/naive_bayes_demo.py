"""
朴素贝叶斯（Naive Bayes）算法的scikit-learn实现
使用鸢尾花数据集进行多分类
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()

X = iris.data  # 所有特征
y = iris.target  # 所有标签

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 2. 数据预处理 - 特征标准化（对于高斯朴素贝叶斯不是必须的，但可以提高性能）
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# 4. 训练和评估函数
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    """训练模型并评估性能"""
    print(f"\n训练{model_name}模型...")
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name}准确率: {accuracy:.2f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    return accuracy

# 5. 测试不同的朴素贝叶斯模型
accuracies = {}

# 高斯朴素贝叶斯（适用于连续特征）
gaussian_model = GaussianNB()
accuracies['高斯朴素贝叶斯'] = train_and_evaluate('高斯朴素贝叶斯', gaussian_model, X_train, y_train, X_test, y_test)

# 多项式朴素贝叶斯（适用于离散特征，如词频）
# 对于多项式朴素贝叶斯，特征值必须非负
X_train_non_negative = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_test_non_negative = (X_test - X_test.min()) / (X_test.max() - X_test.min())

multinomial_model = MultinomialNB()
accuracies['多项式朴素贝叶斯'] = train_and_evaluate('多项式朴素贝叶斯', multinomial_model, X_train_non_negative, y_train, X_test_non_negative, y_test)

# 伯努利朴素贝叶斯（适用于二元特征）
bernoulli_model = BernoulliNB()
accuracies['伯努利朴素贝叶斯'] = train_and_evaluate('伯努利朴素贝叶斯', bernoulli_model, X_train, y_train, X_test, y_test)

# 6. 模型性能对比
print("\n模型性能对比:")
for model_name, acc in accuracies.items():
    print(f"{model_name}: {acc:.2f}")

# 7. 查看模型参数
print("\n高斯朴素贝叶斯模型参数:")
print(f" 类别先验概率: {gaussian_model.class_prior_}")
print(f" 类别计数: {gaussian_model.class_count_}")
print(f" 每个特征的均值: {gaussian_model.theta_}")
print(f" 每个特征的方差: {gaussian_model.sigma_}")

# 8. 简单的预测示例
print("\n预测示例:")
# 随机选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
actual_label = y_test[sample_idx]

# 使用高斯朴素贝叶斯预测
predicted_label = gaussian_model.predict([sample])[0]
print(f"样本特征: {sample}")
print(f"实际类别: {iris.target_names[actual_label]}")
print(f"预测类别: {iris.target_names[predicted_label]}")

# 9. 预测概率
print("\n预测概率:")
pred_proba = gaussian_model.predict_proba([sample])[0]
for i, (class_name, prob) in enumerate(zip(iris.target_names, pred_proba)):
    print(f"{class_name}: {prob:.4f}")

# 10. 可视化不同特征值对预测的影响
print("\n可视化不同特征值对预测的影响...")
# 选择一个特征进行分析
feature_idx = 2  # petal length
feature_name = iris.feature_names[feature_idx]

# 创建特征值范围
feature_min = X_scaled[:, feature_idx].min()
feature_max = X_scaled[:, feature_idx].max()
feature_values = np.linspace(feature_min, feature_max, 100)

# 固定其他特征为均值
mean_features = X_scaled.mean(axis=0)

# 计算不同特征值下的预测概率
probabilities = []
for val in feature_values:
    test_sample = mean_features.copy()
    test_sample[feature_idx] = val
    prob = gaussian_model.predict_proba([test_sample])[0]
    probabilities.append(prob)

probabilities = np.array(probabilities)

# 绘制概率曲线
plt.figure(figsize=(12, 6))
for i, class_name in enumerate(iris.target_names):
    plt.plot(feature_values, probabilities[:, i], label=class_name)
plt.xlabel(f'{feature_name} (标准化)')
plt.ylabel('预测概率')
plt.title(f'{feature_name}对预测的影响')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('naive_bayes_feature_effect.png')
print("特征影响可视化已保存为 'naive_bayes_feature_effect.png'")

print("\nscikit-learn 朴素贝叶斯 Demo完成！")
