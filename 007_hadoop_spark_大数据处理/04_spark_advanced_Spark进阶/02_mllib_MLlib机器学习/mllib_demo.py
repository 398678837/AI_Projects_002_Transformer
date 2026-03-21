#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLlib 机器学习演示

本脚本演示 Spark MLlib 的基本概念、操作和使用方法。
"""

import os
import sys

print("MLlib 机器学习演示")
print("=" * 50)

# 1. MLlib 基本概念
def mllib_basics():
    print("\n1. MLlib 基本概念:")
    print("- MLlib 是 Spark 的机器学习库")
    print("- 提供了丰富的机器学习算法和工具")
    print("- 支持分类、回归、聚类、推荐等多种机器学习任务")
    print("- 提供了两种 API: RDD-based API 和 DataFrame-based API (Spark ML)")
    print("- 设计用于大规模机器学习")

# 2. Spark ML 与 MLlib 的区别
def spark_ml_vs_mllib():
    print("\n2. Spark ML 与 MLlib 的区别:")
    print("- Spark ML:")
    print("  - 基于 DataFrame API")
    print("  - 提供 Pipeline API，支持机器学习工作流")
    print("  - 更高级、更灵活")
    print("  - 推荐使用")
    print("- MLlib:")
    print("  - 基于 RDD API")
    print("  - 较低级、更底层")
    print("  - 逐渐被 Spark ML 取代")

# 3. 数据准备
def data_preparation():
    print("\n3. 数据准备:")
    print("- 读取数据:")
    print("  from pyspark.sql import SparkSession")
    print("  spark = SparkSession.builder.appName('MLlibDemo').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)")
    print("- 数据探索:")
    print("  df.show()")
    print("  df.describe().show()")
    print("  df.printSchema()")
    print("- 数据转换:")
    print("  from pyspark.ml.feature import VectorAssembler")
    print("  assembler = VectorAssembler(inputCols=['feature1', 'feature2'], outputCol='features')")
    print("  df = assembler.transform(df)")

# 4. 特征工程
def feature_engineering():
    print("\n4. 特征工程:")
    print("- 特征提取:")
    print("  # TF-IDF")
    print("  from pyspark.ml.feature import Tokenizer, HashingTF, IDF")
    print("  tokenizer = Tokenizer(inputCol='text', outputCol='words')")
    print("  hashingTF = HashingTF(inputCol='words', outputCol='rawFeatures', numFeatures=1000)")
    print("  idf = IDF(inputCol='rawFeatures', outputCol='features')")
    print("- 特征转换:")
    print("  # 标准化")
    print("  from pyspark.ml.feature import StandardScaler")
    print("  scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')")
    print("  # 独热编码")
    print("  from pyspark.ml.feature import OneHotEncoder")
    print("  encoder = OneHotEncoder(inputCol='category', outputCol='categoryVec')")
    print("- 特征选择:")
    print("  from pyspark.ml.feature import ChiSqSelector")
    print("  selector = ChiSqSelector(numTopFeatures=5, featuresCol='features', outputCol='selectedFeatures', labelCol='label')")

# 5. 分类算法
def classification_algorithms():
    print("\n5. 分类算法:")
    print("- 逻辑回归:")
    print("  from pyspark.ml.classification import LogisticRegression")
    print("  lr = LogisticRegression(maxIter=10, regParam=0.01)")
    print("  model = lr.fit(train_df)")
    print("- 决策树分类:")
    print("  from pyspark.ml.classification import DecisionTreeClassifier")
    print("  dt = DecisionTreeClassifier(labelCol='label', featuresCol='features')")
    print("  model = dt.fit(train_df)")
    print("- 随机森林分类:")
    print("  from pyspark.ml.classification import RandomForestClassifier")
    print("  rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=10)")
    print("  model = rf.fit(train_df)")
    print("- 梯度提升树分类:")
    print("  from pyspark.ml.classification import GBTClassifier")
    print("  gbt = GBTClassifier(labelCol='label', featuresCol='features', maxIter=10)")
    print("  model = gbt.fit(train_df)")

# 6. 回归算法
def regression_algorithms():
    print("\n6. 回归算法:")
    print("- 线性回归:")
    print("  from pyspark.ml.regression import LinearRegression")
    print("  lr = LinearRegression(maxIter=10, regParam=0.01)")
    print("  model = lr.fit(train_df)")
    print("- 决策树回归:")
    print("  from pyspark.ml.regression import DecisionTreeRegressor")
    print("  dt = DecisionTreeRegressor(labelCol='label', featuresCol='features')")
    print("  model = dt.fit(train_df)")
    print("- 随机森林回归:")
    print("  from pyspark.ml.regression import RandomForestRegressor")
    print("  rf = RandomForestRegressor(labelCol='label', featuresCol='features', numTrees=10)")
    print("  model = rf.fit(train_df)")
    print("- 梯度提升树回归:")
    print("  from pyspark.ml.regression import GBTRegressor")
    print("  gbt = GBTRegressor(labelCol='label', featuresCol='features', maxIter=10)")
    print("  model = gbt.fit(train_df)")

# 7. 聚类算法
def clustering_algorithms():
    print("\n7. 聚类算法:")
    print("- K-means:")
    print("  from pyspark.ml.clustering import KMeans")
    print("  kmeans = KMeans(k=3, seed=42)")
    print("  model = kmeans.fit(df)")
    print("- 高斯混合模型:")
    print("  from pyspark.ml.clustering import GaussianMixture")
    print("  gmm = GaussianMixture(k=3, seed=42)")
    print("  model = gmm.fit(df)")
    print("- 分层聚类:")
    print("  from pyspark.ml.clustering import BisectingKMeans")
    print("  bkm = BisectingKMeans(k=3, seed=42)")
    print("  model = bkm.fit(df)")

# 8. 推荐系统
def recommendation_systems():
    print("\n8. 推荐系统:")
    print("- ALS (交替最小二乘):")
    print("  from pyspark.ml.recommendation import ALS")
    print("  als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')")
    print("  model = als.fit(train_df)")
    print("- 生成推荐:")
    print("  # 为每个用户推荐 10 个物品")
    print("  userRecs = model.recommendForAllUsers(10)")
    print("  # 为每个物品推荐 10 个用户")
    print("  itemRecs = model.recommendForAllItems(10)")

# 9. 评估指标
def evaluation_metrics():
    print("\n9. 评估指标:")
    print("- 分类评估:")
    print("  from pyspark.ml.evaluation import MulticlassClassificationEvaluator")
    print("  evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')")
    print("  accuracy = evaluator.evaluate(predictions)")
    print("- 回归评估:")
    print("  from pyspark.ml.evaluation import RegressionEvaluator")
    print("  evaluator = RegressionEvaluator(labelCol='label', predictionCol='prediction', metricName='rmse')")
    print("  rmse = evaluator.evaluate(predictions)")
    print("- 聚类评估:")
    print("  from pyspark.ml.evaluation import ClusteringEvaluator")
    print("  evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette')")
    print("  silhouette = evaluator.evaluate(predictions)")

# 10. 机器学习工作流
def machine_learning_pipeline():
    print("\n10. 机器学习工作流:")
    print("- 创建 Pipeline:")
    print("  from pyspark.ml import Pipeline")
    print("  from pyspark.ml.feature import Tokenizer, HashingTF, IDF")
    print("  from pyspark.ml.classification import LogisticRegression")
    print("  ")
    print("  # 定义 stages")
    print("  tokenizer = Tokenizer(inputCol='text', outputCol='words')")
    print("  hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='rawFeatures')")
    print("  idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol='features')")
    print("  lr = LogisticRegression(maxIter=10, regParam=0.01)")
    print("  ")
    print("  # 创建 Pipeline")
    print("  pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lr])")
    print("  ")
    print("  # 训练模型")
    print("  model = pipeline.fit(train_df)")
    print("  ")
    print("  # 预测")
    print("  predictions = model.transform(test_df)")
    print("  ")
    print("  # 评估")
    print("  from pyspark.ml.evaluation import BinaryClassificationEvaluator")
    print("  evaluator = BinaryClassificationEvaluator()")
    print("  accuracy = evaluator.evaluate(predictions)")
    print("  print(f'Accuracy: {accuracy}")")

if __name__ == "__main__":
    # 执行所有演示
    mllib_basics()
    spark_ml_vs_mllib()
    data_preparation()
    feature_engineering()
    classification_algorithms()
    regression_algorithms()
    clustering_algorithms()
    recommendation_systems()
    evaluation_metrics()
    machine_learning_pipeline()
    
    print("\n" + "=" * 50)
    print("演示完成！")