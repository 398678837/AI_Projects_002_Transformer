# MLlib 机器学习详细文档

## 1. MLlib 基本概念

MLlib 是 Spark 的机器学习库，提供了丰富的机器学习算法和工具，支持分类、回归、聚类、推荐等多种机器学习任务。它提供了两种 API: RDD-based API 和 DataFrame-based API (Spark ML)，设计用于大规模机器学习。

### 1.1 核心特性
- **分布式计算**：利用 Spark 的分布式计算能力处理大规模数据
- **丰富的算法**：支持分类、回归、聚类、推荐等多种机器学习任务
- **Pipeline API**：支持构建端到端的机器学习工作流
- **特征工程**：提供丰富的特征提取、转换和选择工具
- **模型评估**：提供多种评估指标和工具
- **模型持久化**：支持模型的保存和加载

### 1.2 适用场景
- **大规模机器学习**：处理 PB 级数据集
- **分布式模型训练**：利用集群资源加速模型训练
- **端到端机器学习工作流**：从数据准备到模型部署的完整流程
- **实时机器学习**：结合 Spark Streaming 进行实时模型训练和预测

## 2. Spark ML 与 MLlib 的区别

### 2.1 Spark ML
- **基于 DataFrame API**：使用更高级的 DataFrame 作为数据表示
- **Pipeline API**：支持构建和管理机器学习工作流
- **更高级、更灵活**：提供更简洁、更一致的 API
- **推荐使用**：是 Spark 机器学习的推荐选择

### 2.2 MLlib
- **基于 RDD API**：使用较低级的 RDD 作为数据表示
- **较低级、更底层**：API 更复杂，使用起来不够直观
- **逐渐被 Spark ML 取代**：Spark 官方推荐使用 Spark ML

### 2.3 选择建议
- **新项目**：使用 Spark ML
- **现有项目**：如果使用的是 MLlib，可以考虑迁移到 Spark ML
- **特定场景**：某些特定算法可能只在 MLlib 中提供

## 3. 数据准备

### 3.1 读取数据
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('MLlibDemo').getOrCreate()

# 读取 CSV 文件
df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)

# 读取 Parquet 文件
df = spark.read.parquet('hdfs://path/to/data.parquet')

# 读取 JSON 文件
df = spark.read.json('hdfs://path/to/data.json')

# 读取 JDBC 数据
df = spark.read.format('jdbc') \
    .option('url', 'jdbc:mysql://localhost:3306/db') \
    .option('dbtable', 'table') \
    .option('user', 'username') \
    .option('password', 'password') \
    .load()
```

### 3.2 数据探索
```python
# 查看数据前几行
df.show()

# 查看数据统计信息
df.describe().show()

# 查看数据 schema
df.printSchema()

# 查看列名
df.columns

# 查看数据行数
df.count()

# 查看唯一值
df.select('category').distinct().show()
```

### 3.3 数据转换
```python
# 选择特定列
df = df.select('feature1', 'feature2', 'label')

# 过滤数据
df = df.filter(df['feature1'] > 0)

# 处理缺失值
df = df.fillna({'feature1': 0, 'feature2': 0})

# 转换数据类型
df = df.withColumn('feature1', df['feature1'].cast('double'))

# 重命名列
df = df.withColumnRenamed('old_name', 'new_name')
```

### 3.4 特征向量化
```python
from pyspark.ml.feature import VectorAssembler

# 将多个特征列组合成一个特征向量列
assembler = VectorAssembler(
    inputCols=['feature1', 'feature2', 'feature3'],
    outputCol='features'
)

# 转换数据
df = assembler.transform(df)

# 查看结果
df.select('features', 'label').show()
```

## 4. 特征工程

### 4.1 特征提取

#### 4.1.1 TF-IDF
```python
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# 分词
tokenizer = Tokenizer(inputCol='text', outputCol='words')
df = tokenizer.transform(df)

# 计算词频
hashingTF = HashingTF(
    inputCol='words',
    outputCol='rawFeatures',
    numFeatures=1000
)
df = hashingTF.transform(df)

# 计算逆文档频率
idf = IDF(inputCol='rawFeatures', outputCol='features')
idfModel = idf.fit(df)
df = idfModel.transform(df)
```

#### 4.1.2 Word2Vec
```python
from pyspark.ml.feature import Word2Vec

# 训练 Word2Vec 模型
word2Vec = Word2Vec(
    vectorSize=100,
    minCount=5,
    inputCol='words',
    outputCol='word2vec'
)
model = word2Vec.fit(df)
df = model.transform(df)
```

#### 4.1.3 CountVectorizer
```python
from pyspark.ml.feature import CountVectorizer

# 训练 CountVectorizer 模型
cv = CountVectorizer(
    inputCol='words',
    outputCol='features',
    vocabSize=1000,
    minDF=5
)
model = cv.fit(df)
df = model.transform(df)
```

### 4.2 特征转换

#### 4.2.1 标准化
```python
from pyspark.ml.feature import StandardScaler

# 训练标准化模型
scaler = StandardScaler(
    inputCol='features',
    outputCol='scaledFeatures',
    withStd=True,
    withMean=False
)
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)
```

#### 4.2.2 归一化
```python
from pyspark.ml.feature import MinMaxScaler

# 训练归一化模型
scaler = MinMaxScaler(
    inputCol='features',
    outputCol='scaledFeatures'
)
scalerModel = scaler.fit(df)
df = scalerModel.transform(df)
```

#### 4.2.3 独热编码
```python
from pyspark.ml.feature import OneHotEncoder, StringIndexer

# 先将字符串转换为索引
indexer = StringIndexer(
    inputCol='category',
    outputCol='categoryIndex'
)
indexed = indexer.fit(df).transform(df)

# 进行独热编码
encoder = OneHotEncoder(
    inputCol='categoryIndex',
    outputCol='categoryVec'
)
encoded = encoder.fit(indexed).transform(indexed)
```

#### 4.2.4 特征交叉
```python
from pyspark.ml.feature import VectorAssembler

# 组合多个特征
assembler = VectorAssembler(
    inputCols=['feature1', 'feature2', 'categoryVec'],
    outputCol='features'
)
df = assembler.transform(df)
```

### 4.3 特征选择

#### 4.3.1 卡方选择器
```python
from pyspark.ml.feature import ChiSqSelector

# 训练卡方选择器
selector = ChiSqSelector(
    numTopFeatures=5,
    featuresCol='features',
    outputCol='selectedFeatures',
    labelCol='label'
)
selectorModel = selector.fit(df)
df = selectorModel.transform(df)
```

#### 4.3.2 随机森林选择器
```python
from pyspark.ml.feature import RFormula

# 使用 RFormula 选择特征
formula = RFormula(
    formula='label ~ feature1 + feature2 + feature3',
    featuresCol='features',
    labelCol='label'
)
df = formula.fit(df).transform(df)
```

## 5. 分类算法

### 5.1 逻辑回归
```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 分割数据
train_df, test_df = df.randomSplit([0.7, 0.3])

# 创建逻辑回归模型
lr = LogisticRegression(
    maxIter=10,
    regParam=0.01,
    elasticNetParam=0.8,
    labelCol='label',
    featuresCol='features'
)

# 训练模型
model = lr.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
 evaluator = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='accuracy'
)
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# 超参数调优
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

crossval = CrossValidator(
    estimator=lr,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

cvModel = crossval.fit(train_df)
predictions = cvModel.transform(test_df)
accuracy = evaluator.evaluate(predictions)
print(f'Cross-validated Accuracy: {accuracy}')
```

### 5.2 决策树分类
```python
from pyspark.ml.classification import DecisionTreeClassifier

# 创建决策树分类器
dt = DecisionTreeClassifier(
    labelCol='label',
    featuresCol='features',
    maxDepth=5,
    maxBins=32
)

# 训练模型
model = dt.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

### 5.3 随机森林分类
```python
from pyspark.ml.classification import RandomForestClassifier

# 创建随机森林分类器
rf = RandomForestClassifier(
    labelCol='label',
    featuresCol='features',
    numTrees=10,
    maxDepth=5
)

# 训练模型
model = rf.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

### 5.4 梯度提升树分类
```python
from pyspark.ml.classification import GBTClassifier

# 创建梯度提升树分类器
gbt = GBTClassifier(
    labelCol='label',
    featuresCol='features',
    maxIter=10,
    maxDepth=5
)

# 训练模型
model = gbt.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

## 6. 回归算法

### 6.1 线性回归
```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# 创建线性回归模型
lr = LinearRegression(
    maxIter=10,
    regParam=0.01,
    elasticNetParam=0.8,
    labelCol='label',
    featuresCol='features'
)

# 训练模型
model = lr.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
evaluator = RegressionEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='rmse'
)
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 查看系数和截距
print(f'Coefficients: {model.coefficients}')
print(f'Intercept: {model.intercept}')
```

### 6.2 决策树回归
```python
from pyspark.ml.regression import DecisionTreeRegressor

# 创建决策树回归器
dt = DecisionTreeRegressor(
    labelCol='label',
    featuresCol='features',
    maxDepth=5,
    maxBins=32
)

# 训练模型
model = dt.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

### 6.3 随机森林回归
```python
from pyspark.ml.regression import RandomForestRegressor

# 创建随机森林回归器
rf = RandomForestRegressor(
    labelCol='label',
    featuresCol='features',
    numTrees=10,
    maxDepth=5
)

# 训练模型
model = rf.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

### 6.4 梯度提升树回归
```python
from pyspark.ml.regression import GBTRegressor

# 创建梯度提升树回归器
gbt = GBTRegressor(
    labelCol='label',
    featuresCol='features',
    maxIter=10,
    maxDepth=5
)

# 训练模型
model = gbt.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 查看特征重要性
print('Feature Importance:')
for i, imp in enumerate(model.featureImportances):
    print(f'Feature {i}: {imp}')
```

## 7. 聚类算法

### 7.1 K-means
```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# 创建 K-means 模型
kmeans = KMeans(
    k=3,
    seed=42,
    featuresCol='features'
)

# 训练模型
model = kmeans.fit(df)

# 预测
predictions = model.transform(df)

# 评估
evaluator = ClusteringEvaluator(
    predictionCol='prediction',
    featuresCol='features',
    metricName='silhouette'
)
silhouette = evaluator.evaluate(predictions)
print(f'Silhouette Score: {silhouette}')

# 查看聚类中心
centers = model.clusterCenters()
print('Cluster Centers:')
for i, center in enumerate(centers):
    print(f'Cluster {i}: {center}')
```

### 7.2 高斯混合模型
```python
from pyspark.ml.clustering import GaussianMixture

# 创建高斯混合模型
gmm = GaussianMixture(
    k=3,
    seed=42,
    featuresCol='features'
)

# 训练模型
model = gmm.fit(df)

# 预测
predictions = model.transform(df)

# 查看模型参数
print('Weights:')
print(model.weights)
print('Means:')
print(model.gaussiansDF.show())
```

### 7.3 分层聚类
```python
from pyspark.ml.clustering import BisectingKMeans

# 创建分层聚类模型
bkm = BisectingKMeans(
    k=3,
    seed=42,
    featuresCol='features'
)

# 训练模型
model = bkm.fit(df)

# 预测
predictions = model.transform(df)

# 评估
silhouette = evaluator.evaluate(predictions)
print(f'Silhouette Score: {silhouette}')

# 查看聚类中心
centers = model.clusterCenters()
print('Cluster Centers:')
for i, center in enumerate(centers):
    print(f'Cluster {i}: {center}')
```

## 8. 推荐系统

### 8.1 ALS (交替最小二乘)
```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# 分割数据
train_df, test_df = df.randomSplit([0.7, 0.3])

# 创建 ALS 模型
als = ALS(
    maxIter=5,
    regParam=0.01,
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    coldStartStrategy='drop'
)

# 训练模型
model = als.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction'
)
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 为每个用户推荐 10 个物品
userRecs = model.recommendForAllUsers(10)
userRecs.show()

# 为每个物品推荐 10 个用户
itemRecs = model.recommendForAllItems(10)
itemRecs.show()

# 为特定用户推荐物品
specificUser = spark.createDataFrame([(1,)], ['userId'])
specificUserRecs = model.recommendForUserSubset(specificUser, 10)
specificUserRecs.show()
```

## 9. 评估指标

### 9.1 分类评估
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# 多分类评估
evaluator = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='accuracy'
)
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')

# 精确率
evaluator.setMetricName('precisionByLabel')
precision = evaluator.evaluate(predictions)
print(f'Precision: {precision}')

# 召回率
evaluator.setMetricName('recallByLabel')
recall = evaluator.evaluate(predictions)
print(f'Recall: {recall}')

# F1 分数
evaluator.setMetricName('f1')
f1 = evaluator.evaluate(predictions)
print(f'F1 Score: {f1}')

# 二分类评估
binaryEvaluator = BinaryClassificationEvaluator(
    labelCol='label',
    rawPredictionCol='rawPrediction',
    metricName='areaUnderROC'
)
auc = binaryEvaluator.evaluate(predictions)
print(f'AUC: {auc}')
```

### 9.2 回归评估
```python
from pyspark.ml.evaluation import RegressionEvaluator

# 均方根误差 (RMSE)
evaluator = RegressionEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='rmse'
)
rmse = evaluator.evaluate(predictions)
print(f'RMSE: {rmse}')

# 平均绝对误差 (MAE)
evaluator.setMetricName('mae')
mae = evaluator.evaluate(predictions)
print(f'MAE: {mae}')

# R² 评分
evaluator.setMetricName('r2')
r2 = evaluator.evaluate(predictions)
print(f'R²: {r2}')

# 均方误差 (MSE)
evaluator.setMetricName('mse')
mse = evaluator.evaluate(predictions)
print(f'MSE: {mse}')
```

### 9.3 聚类评估
```python
from pyspark.ml.evaluation import ClusteringEvaluator

# 轮廓系数
evaluator = ClusteringEvaluator(
    predictionCol='prediction',
    featuresCol='features',
    metricName='silhouette'
)
silhouette = evaluator.evaluate(predictions)
print(f'Silhouette Score: {silhouette}')
```

## 10. 机器学习工作流

### 10.1 创建 Pipeline
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# 定义 stages
tokenizer = Tokenizer(inputCol='text', outputCol='words')
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='rawFeatures')
idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol='features')
lr = LogisticRegression(maxIter=10, regParam=0.01)

# 创建 Pipeline
pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])

# 训练模型
model = pipeline.fit(train_df)

# 预测
predictions = model.transform(test_df)

# 评估
evaluator = BinaryClassificationEvaluator()
accuracy = evaluator.evaluate(predictions)
print(f'Accuracy: {accuracy}')
```

### 10.2 超参数调优
```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 创建参数网格
paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [1000, 2000]) \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.maxIter, [10, 20]) \
    .build()

# 创建交叉验证器
crossval = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3
)

# 训练模型
cvModel = crossval.fit(train_df)

# 预测
predictions = cvModel.transform(test_df)

# 评估
accuracy = evaluator.evaluate(predictions)
print(f'Cross-validated Accuracy: {accuracy}')

# 查看最佳参数
bestModel = cvModel.bestModel
print('Best Model Parameters:')
print(f'  HashingTF numFeatures: {bestModel.stages[1].getNumFeatures()}')
print(f'  LogisticRegression regParam: {bestModel.stages[3].getRegParam()}')
print(f'  LogisticRegression maxIter: {bestModel.stages[3].getMaxIter()}')
```

### 10.3 模型持久化
```python
# 保存模型
model.save('hdfs://path/to/model')

# 加载模型
from pyspark.ml import PipelineModel
loaded_model = PipelineModel.load('hdfs://path/to/model')

# 使用加载的模型进行预测
predictions = loaded_model.transform(test_df)
```

## 11. 最佳实践

### 11.1 数据处理
- **数据清洗**：处理缺失值、异常值和重复数据
- **特征工程**：根据业务需求选择和转换特征
- **数据分区**：合理设置数据分区，提高并行度
- **数据缓存**：对频繁使用的数据进行缓存

### 11.2 模型选择
- **算法选择**：根据任务类型和数据特点选择合适的算法
- **超参数调优**：使用交叉验证和网格搜索调优超参数
- **模型评估**：使用合适的评估指标评估模型性能
- **模型集成**：考虑使用模型集成提高性能

### 11.3 性能优化
- **资源配置**：根据数据量和模型复杂度调整资源配置
- **数据序列化**：使用 Kryo 序列化提高性能
- **内存管理**：合理设置内存分配，避免 OOM
- **并行度**：根据集群资源调整并行度

### 11.4 部署建议
- **模型持久化**：保存训练好的模型，避免重复训练
- **批处理**：使用 Spark 批处理进行离线模型训练
- **流处理**：结合 Spark Streaming 进行实时模型预测
- **模型监控**：监控模型性能，定期更新模型

### 11.5 常见问题
- **数据倾斜**：使用数据预处理和分区策略解决数据倾斜
- **内存溢出**：调整内存配置和数据分区
- **训练速度慢**：使用更高效的算法和优化资源配置
- **过拟合**：使用正则化、交叉验证等方法防止过拟合

## 12. 总结

MLlib 是 Spark 生态系统中用于机器学习的重要库，提供了丰富的算法和工具，支持大规模机器学习任务。本文档详细介绍了 MLlib 的基本概念、Spark ML 与 MLlib 的区别、数据准备、特征工程、分类算法、回归算法、聚类算法、推荐系统、评估指标和机器学习工作流。

Spark ML 作为 MLlib 的继任者，提供了更高级、更灵活的 API，特别是 Pipeline API 支持构建端到端的机器学习工作流。通过合理使用 Spark ML 的各种特性，可以构建高效、可靠的机器学习应用。

掌握 MLlib 的使用技巧，对于深入理解机器学习和构建大规模机器学习应用至关重要。通过本文档的学习，您应该能够在实际应用中灵活运用 MLlib 处理和分析数据，构建和部署机器学习模型。