"""
LDA（线性判别分析）降维算法的scikit-learn实现
使用鸢尾花数据集进行降维和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 加载数据集
print("加载鸢尾花数据集...")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征名称: {feature_names}")
print(f"Class名称: {target_names}")

# 2. 数据预处理 - 特征标准化
print("\n特征标准化...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 应用LDA降维
print("\n应用LDA降维...")

# 创建LDA模型
# 注意：LDA最多只能降到n_classes-1维
lda = LinearDiscriminantAnalysis(n_components=2)

# 训练模型并降维
X_lda = lda.fit_transform(X_scaled, y)

print(f"降维后数据形状: {X_lda.shape}")
print(f"解释方差比: {lda.explained_variance_ratio_}")
print(f"累计解释方差比: {np.sum(lda.explained_variance_ratio_):.4f}")

# 4. 可视化降维结果
print("\n可视化降维结果...")

plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA Dimensionality Reduction - Iris Dataset')
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/lda_iris_visualization.png')
print("LDA可视化已保存为 'images/lda_iris_visualization.png'")

# 5. 与PCA对比
print("\n与PCA对比...")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA累计解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")

# 绘制对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PCA
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
ax1.set_title('PCA降维')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# LDA
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax2.scatter(X_lda[y == i, 0], X_lda[y == i, 1], 
                color=color, alpha=.8, lw=lw, label=target_name)
ax2.set_title('LDA降维')
ax2.set_xlabel('LDA 1')
ax2.set_ylabel('LDA 2')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/lda_vs_pca_comparison.png')
print("PCA与LDA对比已保存为 'images/lda_vs_pca_comparison.png'")

# 6. LDA用于分类
print("\n使用LDA进行分类...")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 使用LDA进行分类
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train, y_train)
y_pred = lda_classifier.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"LDA分类Accuracy: {accuracy:.4f}")

# 7. LDA参数说明
print("\nLDA参数说明:")
print("- n_components: 降维后的维度")
print("  - 最多只能降到n_classes-1维")
print("  - 鸢尾花数据集有3个Class，所以最多降到2维")
print()
print("- solver: 求解器")
print("  - 'svd': 奇异值分解（默认），不计算协方差矩阵")
print("  - 'lsqr': 最小平方，结合正则化")
print("  - 'eigen': 特征值分解，结合正则化")
print()
print("- shrinkage: 收缩参数，用于正则化")
print("  - None: 不使用收缩（默认）")
print("  - 'auto': 自动使用Ledoit-Wolf收缩")
print("  - float: 0到1之间的浮点数")

print("\nLDA vs PCA:")
print("- LDA: 有监督降维，最大化Class间距离")
print("- PCA: 无监督降维，最大化方差")
print("- LDA通常在分类任务上表现更好")

print("\nscikit-learn LDA Demo完成！")
