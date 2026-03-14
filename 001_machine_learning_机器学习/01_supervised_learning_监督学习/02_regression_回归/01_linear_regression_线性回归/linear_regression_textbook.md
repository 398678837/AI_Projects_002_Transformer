# 线性回归教材

## 第一章：回归分析基础

### 1.1 什么是回归分析

回归分析是一种**预测建模技术**，用于研究因变量（目标变量）与一个或多个自变量（特征变量）之间的关系。回归分析的目标是建立一个数学模型，能够根据自变量的值预测因变量的值。

#### 1.1.1 回归分析的历史
- **起源**：19世纪，由弗朗西斯·高尔顿（Francis Galton）在研究遗传学时提出
- **发展**：卡尔·皮尔逊（Karl Pearson）发展了相关系数
- **现代应用**：广泛应用于经济学、金融学、医学、工程学等领域

#### 1.1.2 回归分析的类型
1. **简单线性回归**：一个自变量，一个因变量
2. **多元线性回归**：多个自变量，一个因变量
3. **多项式回归**：自变量的高次项
4. **非线性回归**：非线性关系

### 1.2 线性回归的数学基础

#### 1.2.1 线性模型
线性回归模型假设因变量y与自变量x之间存在线性关系：

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon $$

其中：
- $y$ 是因变量（目标变量）
- $x_1, x_2, \ldots, x_n$ 是自变量（特征变量）
- $\beta_0$ 是截距
- $\beta_1, \beta_2, \ldots, \beta_n$ 是回归系数
- $\epsilon$ 是误差项

#### 1.2.2 矩阵形式
线性回归可以用矩阵形式表示：

$$ \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon} $$

其中：
- $\mathbf{y}$ 是 $n \times 1$ 的因变量向量
- $\mathbf{X}$ 是 $n \times (p+1)$ 的设计矩阵
- $\boldsymbol{\beta}$ 是 $(p+1) \times 1$ 的回归系数向量
- $\boldsymbol{\epsilon}$ 是 $n \times 1$ 的误差向量

## 第二章：最小二乘法

### 2.1 最小二乘法的原理

最小二乘法（Ordinary Least Squares, OLS）是估计线性回归参数最常用的方法。其基本思想是**最小化残差平方和**。

#### 2.1.1 残差
残差是实际值与预测值之间的差异：

$$ e_i = y_i - \hat{y}_i = y_i - (\beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}) $$

#### 2.1.2 目标函数
最小二乘法的目标是最小化残差平方和：

$$ S(\boldsymbol{\beta}) = \sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

### 2.2 正规方程

#### 2.2.1 推导过程
对目标函数求偏导并令其为零：

$$ \frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0 $$

解得：

$$ \boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} $$

#### 2.2.2 几何解释
最小二乘法的几何意义是找到一条直线，使得所有数据点到这条直线的垂直距离（残差）的平方和最小。

### 2.3 梯度下降法

#### 2.3.1 梯度下降的原理
当数据量很大时，直接求解正规方程计算量太大，可以使用梯度下降法：

$$ \boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \alpha \nabla S(\boldsymbol{\beta}^{(t)}) $$

其中：
- $\alpha$ 是学习率
- $\nabla S(\boldsymbol{\beta})$ 是目标函数的梯度

#### 2.3.2 梯度计算
$$ \nabla S(\boldsymbol{\beta}) = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) $$

## 第三章：模型评估

### 3.1 评估指标

#### 3.1.1 均方误差（MSE）
$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

#### 3.1.2 均方根误差（RMSE）
$$ RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

#### 3.1.3 平均绝对误差（MAE）
$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

#### 3.1.4 决定系数（R²）
$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

其中：
- $SS_{res}$ 是残差平方和
- $SS_{tot}$ 是总平方和
- $\bar{y}$ 是y的均值

### 3.2 模型诊断

#### 3.2.1 残差分析
- **残差图**：检查残差是否随机分布
- **正态性检验**：检查残差是否服从正态分布
- **异方差性**：检查残差方差是否恒定

#### 3.2.2 多重共线性
- **方差膨胀因子（VIF）**：
  $$ VIF_j = \frac{1}{1 - R_j^2} $$
  其中 $R_j^2$ 是第j个自变量对其他自变量回归的决定系数

- **诊断标准**：VIF > 10 表示存在严重的多重共线性

## 第四章：假设检验

### 4.1 回归系数的显著性检验

#### 4.1.1 t检验
检验单个回归系数是否显著不为零：

$$ t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} $$

其中：
- $\hat{\beta}_j$ 是估计的回归系数
- $SE(\hat{\beta}_j)$ 是标准误

#### 4.1.2 置信区间
回归系数的95%置信区间：

$$ \hat{\beta}_j \pm t_{\alpha/2, n-p-1} \times SE(\hat{\beta}_j) $$

### 4.2 模型的整体显著性检验

#### 4.2.1 F检验
检验模型整体是否显著：

$$ F = \frac{MS_{reg}}{MS_{res}} = \frac{SS_{reg}/p}{SS_{res}/(n-p-1)} $$

其中：
- $MS_{reg}$ 是回归均方
- $MS_{res}$ 是残差均方
- $p$ 是自变量个数

## 第五章：实际应用

### 5.1 数据预处理

#### 5.1.1 缺失值处理
- **删除**：删除包含缺失值的样本
- **填充**：用均值、中位数或众数填充
- **插值**：使用插值方法估计缺失值

#### 5.1.2 异常值处理
- **检测**：使用箱线图、Z-score等方法检测
- **处理**：删除、替换或保留

#### 5.1.3 特征缩放
- **标准化**：$x' = \frac{x - \mu}{\sigma}$
- **归一化**：$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$

### 5.2 特征工程

#### 5.2.1 特征选择
- **相关性分析**：选择与目标变量相关性高的特征
- **逐步回归**：逐步添加或删除特征
- **正则化**：使用L1正则化进行特征选择

#### 5.2.2 特征构造
- **多项式特征**：添加特征的平方、立方等
- **交互特征**：添加特征的乘积
- **对数变换**：对偏态分布进行对数变换

### 5.3 模型选择

#### 5.3.1 交叉验证
- **K折交叉验证**：将数据分成K份，轮流作为测试集
- **留一法**：每次留一个样本作为测试集

#### 5.3.2 信息准则
- **AIC（赤池信息准则）**：
  $$ AIC = 2k - 2\ln(\hat{L}) $$
  其中 $k$ 是参数个数，$\hat{L}$ 是似然函数的最大值

- **BIC（贝叶斯信息准则）**：
  $$ BIC = k\ln(n) - 2\ln(\hat{L}) $$

## 第六章：扩展方法

### 6.1 加权最小二乘法

当误差项的方差不恒定时，可以使用加权最小二乘法：

$$ S(\boldsymbol{\beta}) = \sum_{i=1}^{n} w_i (y_i - \hat{y}_i)^2 $$

其中 $w_i$ 是权重。

### 6.2 稳健回归

当数据中存在异常值时，可以使用稳健回归方法：
- **Huber回归**：结合最小二乘和最小绝对偏差
- **RANSAC**：随机抽样一致性

### 6.3 广义线性模型

当因变量不服从正态分布时，可以使用广义线性模型：
- **逻辑回归**：二分类问题
- **泊松回归**：计数数据
- **伽马回归**：正值连续数据

## 第七章：Python实现

### 7.1 使用NumPy实现

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # 添加截距项
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 正规方程求解
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]
    
    def predict(self, X):
        return self.intercept + X.dot(self.coefficients)
```

### 7.2 使用scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 创建模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## 第八章：案例分析

### 8.1 房价预测

**数据集**：California Housing Dataset

**特征**：
- MedInc：收入中位数
- HouseAge：房屋年龄中位数
- AveRooms：平均房间数
- AveBedrms：平均卧室数
- Population：人口
- AveOccup：平均居住人数
- Latitude：纬度
- Longitude：经度

**目标**：预测房价中位数

**结果**：
- MSE: 0.5559
- RMSE: 0.7456
- R²: 0.5758

### 8.2 销售预测

**数据集**：零售销售数据

**特征**：
- 广告支出
- 促销费用
- 季节因素
- 经济指数

**目标**：预测销售额

**结果**：
- MSE: 2.7831
- RMSE: 1.6683
- R²: 0.6058

## 第九章：常见问题

### 9.1 过拟合与欠拟合

#### 9.1.1 过拟合
- **表现**：训练集表现好，测试集表现差
- **原因**：模型过于复杂，学习了噪声
- **解决**：正则化、交叉验证、增加数据

#### 9.1.2 欠拟合
- **表现**：训练集和测试集表现都差
- **原因**：模型过于简单，未能捕捉数据规律
- **解决**：增加特征、使用更复杂的模型

### 9.2 非线性关系

当数据存在非线性关系时，线性回归可能不适用：
- **多项式回归**：添加多项式特征
- **样条回归**：使用分段多项式
- **核方法**：使用核技巧

### 9.3 异方差性

当误差项的方差不恒定时：
- **加权最小二乘法**：给不同样本不同权重
- **对数变换**：对因变量进行对数变换
- **Box-Cox变换**：使用Box-Cox变换

## 第十章：总结

### 10.1 核心要点

1. **线性回归**是一种简单但强大的预测建模技术
2. **最小二乘法**是估计参数最常用的方法
3. **模型评估**需要使用多种指标综合判断
4. **假设检验**可以验证模型的统计显著性
5. **数据预处理**和**特征工程**对模型性能至关重要

### 10.2 学习路径

1. **基础阶段**：理解线性回归的数学原理
2. **实践阶段**：使用Python实现线性回归
3. **进阶阶段**：学习正则化、特征工程等高级技术
4. **应用阶段**：在实际项目中应用线性回归

### 10.3 进一步学习

- **正则化方法**：岭回归、LASSO、弹性网络
- **广义线性模型**：逻辑回归、泊松回归
- **贝叶斯线性回归**：从贝叶斯角度理解线性回归
- **机器学习集成**：将线性回归与其他算法结合

---

**练习题目**：

1. 证明最小二乘估计量是无偏的。
2. 推导正规方程的矩阵形式。
3. 解释为什么R²可能为负值。
4. 设计一个实验比较不同特征缩放方法的效果。
5. 实现一个带有L2正则化的线性回归模型。

**参考资料**：
- 《统计学习方法》李航
- 《机器学习》周志华
- 《The Elements of Statistical Learning》Hastie, Tibshirani, Friedman
- 《Pattern Recognition and Machine Learning》Bishop
