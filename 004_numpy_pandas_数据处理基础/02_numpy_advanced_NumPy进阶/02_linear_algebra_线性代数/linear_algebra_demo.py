"""
线性代数演示
Linear Algebra Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(script_dir, 'images')

print("=" * 70)
print("线性代数 (Linear Algebra) 演示")
print("=" * 70)

# 1. 基本矩阵运算
print("\n1. 基本矩阵运算...")

# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("矩阵A:")
print(A)
print("\n矩阵B:")
print(B)

# 矩阵加法
print("\n矩阵加法 (A + B):")
print(A + B)

# 矩阵减法
print("\n矩阵减法 (A - B):")
print(A - B)

# 矩阵乘法
print("\n矩阵乘法 (A @ B):")
print(A @ B)

# 标量乘法
print("\n标量乘法 (A * 2):")
print(A * 2)

# 2. 矩阵转置
print("\n2. 矩阵转置...")

print("矩阵A:")
print(A)
print("\n矩阵A的转置:")
print(A.T)

# 3. 矩阵求逆
print("\n3. 矩阵求逆...")

# 计算逆矩阵
A_inv = np.linalg.inv(A)
print("矩阵A的逆:")
print(A_inv)

# 验证逆矩阵
print("\n验证 A @ A_inv = I:")
print(A @ A_inv)

# 4. 行列式
print("\n4. 行列式...")

# 计算行列式
det_A = np.linalg.det(A)
print("矩阵A的行列式:", det_A)

# 5. 特征值和特征向量
print("\n5. 特征值和特征向量...")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)
print("\n特征向量:")
print(eigenvectors)

# 6. 奇异值分解 (SVD)
print("\n6. 奇异值分解 (SVD)...")

# 执行SVD
U, S, Vh = np.linalg.svd(A)
print("U矩阵:")
print(U)
print("\n奇异值:", S)
print("\nVh矩阵:")
print(Vh)

# 7. 求解线性方程组
print("\n7. 求解线性方程组...")

# 定义方程组: 2x + 3y = 8, 5x - y = 3
A = np.array([[2, 3], [5, -1]])
b = np.array([8, 3])

# 求解
x = np.linalg.solve(A, b)
print("方程组的解:", x)

# 验证
print("\n验证 A @ x = b:")
print(A @ x)
print("b:", b)

# 8. 矩阵分解
print("\n8. 矩阵分解...")

# LU分解
from scipy.linalg import lu
P, L, U = lu(A)
print("LU分解:")
print("P矩阵:")
print(P)
print("\nL矩阵:")
print(L)
print("\nU矩阵:")
print(U)

# QR分解
Q, R = np.linalg.qr(A)
print("\nQR分解:")
print("Q矩阵:")
print(Q)
print("\nR矩阵:")
print(R)

# 9. 矩阵范数
print("\n9. 矩阵范数...")

# 计算不同范数
print("矩阵A:")
print(A)
print("\nL1范数:", np.linalg.norm(A, ord=1))
print("L2范数:", np.linalg.norm(A, ord=2))
print("无穷范数:", np.linalg.norm(A, ord=np.inf))
print("Frobenius范数:", np.linalg.norm(A, ord='fro'))

# 10. 矩阵秩
print("\n10. 矩阵秩...")

# 计算矩阵秩
rank = np.linalg.matrix_rank(A)
print("矩阵A的秩:", rank)

# 11. 应用示例
print("\n11. 应用示例...")

# 11.1 线性回归
print("\n11.1 线性回归...")

# 生成样本数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# 构造设计矩阵
X = np.column_stack((np.ones_like(x), x))

# 使用最小二乘法求解
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("回归系数:", beta)

# 预测
y_pred = X @ beta

# 11.2 主成分分析 (PCA)
print("\n11.2 主成分分析 (PCA)...")

# 生成二维数据
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 100)

# 中心化数据
centered_data = data - np.mean(data, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(centered_data.T)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 排序特征值和特征向量
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("特征值:", eigenvalues)
print("\n特征向量:")
print(eigenvectors)

# 12. 性能测试
print("\n12. 性能测试...")
import time

# 测试不同大小矩阵的乘法性能
sizes = [100, 200, 500, 1000]
times = []

for size in sizes:
    # 创建随机矩阵
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    # 测试乘法性能
    start = time.time()
    C = A @ B
    end = time.time()
    times.append(end - start)
    print(f"{size}x{size}矩阵乘法耗时: {end - start:.6f}秒")

# 13. 可视化
print("\n13. 可视化...")

# 创建可视化数据
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 线性回归可视化
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.6, label='数据点')
ax.plot(x, y_pred, color='red', linewidth=2, label='回归线')
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_title('线性回归', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# PCA可视化
ax = axes[0, 1]
ax.scatter(data[:, 0], data[:, 1], alpha=0.6, label='原始数据')

# 绘制主成分
for i in range(len(eigenvalues)):
    eigenvector = eigenvectors[:, i]
    eigenvalue = eigenvalues[i]
    # 缩放特征向量以更好地显示
    scaled_eigenvector = eigenvector * np.sqrt(eigenvalue) * 2
    ax.quiver(0, 0, scaled_eigenvector[0], scaled_eigenvector[1], 
              angles='xy', scale_units='xy', scale=1, 
              color='red', linewidth=2, label=f'主成分{i+1}')

ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_title('主成分分析', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axis('equal')

# 矩阵乘法性能
ax = axes[1, 0]
ax.plot(sizes, times, marker='o', color='steelblue')
ax.set_xlabel('矩阵大小', fontsize=10)
ax.set_ylabel('时间(秒)', fontsize=10)
ax.set_title('矩阵乘法性能', fontsize=12)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# 特征值分布
ax = axes[1, 1]
# 生成随机对称矩阵
np.random.seed(42)
size = 100
A = np.random.rand(size, size)
A = (A + A.T) / 2
# 计算特征值
eigenvalues = np.linalg.eigvals(A)
# 绘制特征值分布
ax.hist(eigenvalues, bins=30, alpha=0.7, color='coral')
ax.set_xlabel('特征值', fontsize=10)
ax.set_ylabel('频率', fontsize=10)
ax.set_title('随机对称矩阵的特征值分布', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 确保images目录存在
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'linear_algebra.png'))
print("可视化已保存为 'images/linear_algebra.png'")

# 14. 总结
print("\n" + "=" * 70)
print("线性代数总结")
print("=" * 70)

print("""
NumPy线性代数功能：

1. 基本矩阵运算：
   - 加法、减法、乘法、标量乘法
   - 矩阵转置

2. 矩阵分解：
   - 逆矩阵
   - 行列式
   - 特征值和特征向量
   - 奇异值分解 (SVD)
   - LU分解
   - QR分解

3. 线性方程组：
   - 求解线性方程组
   - 最小二乘法

4. 矩阵分析：
   - 范数
   - 秩
   - 条件数

5. 应用场景：
   - 线性回归
   - 主成分分析 (PCA)
   - 图像处理
   - 信号处理
   - 机器学习

6. 性能考虑：
   - 大型矩阵运算的性能
   - 算法选择
   - 内存使用
""")

print("=" * 70)
print("线性代数演示完成！")
print("=" * 70)