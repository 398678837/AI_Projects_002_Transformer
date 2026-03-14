"""
高级降维算法（Isomap、LLE、t-SNE）的scikit-learn实现
使用scikit-learn的人工数据集进行降维和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 生成S曲线数据集
print("生成S曲线数据集...")
X_s_curve, y_s_curve = make_s_curve(n_samples=1000, noise=0.1, random_state=42)

print(f"S曲线数据集形状: {X_s_curve.shape}")

# 2. 生成瑞士卷数据集
print("\n生成瑞士卷数据集...")
X_swiss, y_swiss = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

print(f"瑞士卷数据集形状: {X_swiss.shape}")

# 3. 数据预处理
print("\n数据预处理...")
scaler = StandardScaler()
X_s_curve_scaled = scaler.fit_transform(X_s_curve)
X_swiss_scaled = scaler.fit_transform(X_swiss)

# 4. 可视化原始数据
print("\n可视化原始数据...")

fig = plt.figure(figsize=(15, 6))

# S曲线
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter1 = ax1.scatter(X_s_curve[:, 0], X_s_curve[:, 1], X_s_curve[:, 2], 
                        c=y_s_curve, cmap='viridis', s=50)
ax1.set_title('S曲线 - 原始数据')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
plt.colorbar(scatter1, ax=ax1, label='位置')

# 瑞士卷
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
scatter2 = ax2.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], 
                        c=y_swiss, cmap='viridis', s=50)
ax2.set_title('瑞士卷 - 原始数据')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.colorbar(scatter2, ax=ax2, label='位置')

plt.tight_layout()
plt.savefig('images/original_manifold_data.png')
print("原始数据可视化已保存为 'images/original_manifold_data.png'")

# 5. 应用各种降维方法到S曲线
print("\n应用各种降维方法到S曲线...")

# PCA
pca = PCA(n_components=2, random_state=42)
X_s_pca = pca.fit_transform(X_s_curve_scaled)

# Isomap
isomap = Isomap(n_neighbors=10, n_components=2)
X_s_isomap = isomap.fit_transform(X_s_curve_scaled)

# LLE
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard', random_state=42)
X_s_lle = lle.fit_transform(X_s_curve_scaled)

# 6. 可视化S曲线的降维结果
print("\n可视化S曲线的降维结果...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 原始数据（投影到2D）
axes[0, 0].scatter(X_s_curve[:, 0], X_s_curve[:, 2], c=y_s_curve, cmap='viridis', s=50)
axes[0, 0].set_title('S曲线 - 原始数据（2D投影）')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Z')
axes[0, 0].grid(True, alpha=0.3)

# PCA
axes[0, 1].scatter(X_s_pca[:, 0], X_s_pca[:, 1], c=y_s_curve, cmap='viridis', s=50)
axes[0, 1].set_title('S曲线 - PCA降维')
axes[0, 1].set_xlabel('PCA 1')
axes[0, 1].set_ylabel('PCA 2')
axes[0, 1].grid(True, alpha=0.3)

# Isomap
axes[1, 0].scatter(X_s_isomap[:, 0], X_s_isomap[:, 1], c=y_s_curve, cmap='viridis', s=50)
axes[1, 0].set_title('S曲线 - Isomap降维')
axes[1, 0].set_xlabel('Isomap 1')
axes[1, 0].set_ylabel('Isomap 2')
axes[1, 0].grid(True, alpha=0.3)

# LLE
axes[1, 1].scatter(X_s_lle[:, 0], X_s_lle[:, 1], c=y_s_curve, cmap='viridis', s=50)
axes[1, 1].set_title('S曲线 - LLE降维')
axes[1, 1].set_xlabel('LLE 1')
axes[1, 1].set_ylabel('LLE 2')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/s_curve_dimensionality_reduction.png')
print("S曲线降维结果已保存为 'images/s_curve_dimensionality_reduction.png'")

# 7. 应用各种降维方法到瑞士卷
print("\n应用各种降维方法到瑞士卷...")

# PCA
X_swiss_pca = pca.fit_transform(X_swiss_scaled)

# Isomap
X_swiss_isomap = isomap.fit_transform(X_swiss_scaled)

# LLE
X_swiss_lle = lle.fit_transform(X_swiss_scaled)

# 8. 可视化瑞士卷的降维结果
print("\n可视化瑞士卷的降维结果...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 原始数据（投影到2D）
axes[0, 0].scatter(X_swiss[:, 0], X_swiss[:, 2], c=y_swiss, cmap='viridis', s=50)
axes[0, 0].set_title('瑞士卷 - 原始数据（2D投影）')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Z')
axes[0, 0].grid(True, alpha=0.3)

# PCA
axes[0, 1].scatter(X_swiss_pca[:, 0], X_swiss_pca[:, 1], c=y_swiss, cmap='viridis', s=50)
axes[0, 1].set_title('瑞士卷 - PCA降维')
axes[0, 1].set_xlabel('PCA 1')
axes[0, 1].set_ylabel('PCA 2')
axes[0, 1].grid(True, alpha=0.3)

# Isomap
axes[1, 0].scatter(X_swiss_isomap[:, 0], X_swiss_isomap[:, 1], c=y_swiss, cmap='viridis', s=50)
axes[1, 0].set_title('瑞士卷 - Isomap降维')
axes[1, 0].set_xlabel('Isomap 1')
axes[1, 0].set_ylabel('Isomap 2')
axes[1, 0].grid(True, alpha=0.3)

# LLE
axes[1, 1].scatter(X_swiss_lle[:, 0], X_swiss_lle[:, 1], c=y_swiss, cmap='viridis', s=50)
axes[1, 1].set_title('瑞士卷 - LLE降维')
axes[1, 1].set_xlabel('LLE 1')
axes[1, 1].set_ylabel('LLE 2')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/swiss_roll_dimensionality_reduction.png')
print("瑞士卷降维结果已保存为 'images/swiss_roll_dimensionality_reduction.png'")

# 9. 算法说明
print("\n算法说明:")
print("\nPCA:")
print("- 线性降维方法")
print("- 最大化方差")
print("- 不能保持流形结构")

print("\nIsomap:")
print("- 等度量映射")
print("- 保持测地线距离")
print("- 可以展开非线性流形")
print("- 参数: n_neighbors（邻居数量）")

print("\nLLE (Locally Linear Embedding):")
print("- 局部线性嵌入")
print("- 保持局部邻域关系")
print("- 可以展开非线性流形")
print("- 参数: n_neighbors（邻居数量）")

print("\n各方法对比:")
print("- PCA: 快，线性，不能保持流形")
print("- Isomap: 可以保持全局结构，计算慢")
print("- LLE: 可以保持局部结构，对噪声敏感")

print("\nscikit-learn 高级降维算法 Demo完成！")
