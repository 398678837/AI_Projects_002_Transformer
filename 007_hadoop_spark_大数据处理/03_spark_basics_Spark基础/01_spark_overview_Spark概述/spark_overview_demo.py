#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark 概述演示

本脚本演示 Spark 的基本概念、架构和使用方法。
"""

import os
import sys

print("Spark 概述演示")
print("=" * 50)

# 1. Spark 基本概念
def spark_basics():
    print("\n1. Spark 基本概念:")
    print("- Spark 是一个快速、通用的分布式计算引擎")
    print("- 设计用于大规模数据处理")
    print("- 支持批处理、流处理、机器学习、图计算等多种计算模式")
    print("- 比 MapReduce 快 100 倍以上，因为数据缓存在内存中")

# 2. Spark 架构
def spark_architecture():
    print("\n2. Spark 架构:")
    print("- Driver: 驱动程序，负责作业调度和协调")
    print("- Executor: 执行器，运行在工作节点上，执行任务")
    print("- Cluster Manager: 集群管理器，负责资源分配")
    print("- Worker Node: 工作节点，运行 Executor")
    print("- Task: 基本执行单元")
    print("- Job: 由多个 Stage 组成的作业")
    print("- Stage: 由多个 Task 组成的阶段")

# 3. Spark 安装和配置
def spark_installation():
    print("\n3. Spark 安装和配置:")
    print("- 下载 Spark: https://spark.apache.org/downloads.html")
    print("- 解压安装包: tar -xzf spark-3.5.0-bin-hadoop3.tgz")
    print("- 配置环境变量:")
    print("  export SPARK_HOME=/path/to/spark")
    print("  export PATH=$PATH:$SPARK_HOME/bin")
    print("- 主要配置文件:")
    print("  - spark-defaults.conf: 默认配置")
    print("  - spark-env.sh: 环境变量配置")

# 4. Spark 运行模式
def spark_running_modes():
    print("\n4. Spark 运行模式:")
    print("- 本地模式 (Local): 在单机上运行，适合开发和测试")
    print("  spark-submit --master local[*] app.py")
    print("- 集群模式 (Cluster): 在集群上运行")
    print("  - Standalone: Spark 自带的集群管理器")
    print("  - YARN: 运行在 Hadoop YARN 上")
    print("  - Mesos: 运行在 Apache Mesos 上")
    print("  - Kubernetes: 运行在 Kubernetes 上")

# 5. Spark 核心组件
def spark_core_components():
    print("\n5. Spark 核心组件:")
    print("- Spark Core: 核心引擎，提供基本功能")
    print("- Spark SQL: 结构化数据处理")
    print("- Spark Streaming: 流处理")
    print("- MLlib: 机器学习库")
    print("- GraphX: 图计算库")
    print("- Structured Streaming: 结构化流处理")

# 6. Spark 应用提交
def spark_application_submission():
    print("\n6. Spark 应用提交:")
    print("- 提交 Python 应用:")
    print("  spark-submit --master local[*] app.py")
    print("- 提交 JAR 应用:")
    print("  spark-submit --master yarn --class com.example.App app.jar")
    print("- 常用参数:")
    print("  --master: 指定集群管理器")
    print("  --deploy-mode: 部署模式 (client/cluster)")
    print("  --executor-memory: 执行器内存")
    print("  --num-executors: 执行器数量")
    print("  --executor-cores: 执行器核心数")

# 7. Spark 交互式环境
def spark_interactive_environments():
    print("\n7. Spark 交互式环境:")
    print("- Spark Shell: Scala 交互式环境")
    print("  spark-shell")
    print("- PySpark: Python 交互式环境")
    print("  pyspark")
    print("- Spark SQL: SQL 交互式环境")
    print("  spark-sql")
    print("- Jupyter Notebook: 结合 PySpark 使用")
    print("  PYSPARK_DRIVER_PYTHON=jupyter PYSPARK_DRIVER_PYTHON_OPTS='notebook' pyspark")

# 8. Spark 示例应用
def spark_example_applications():
    print("\n8. Spark 示例应用:")
    print("- 单词计数 (WordCount):")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'WordCount')")
    print("  text_file = sc.textFile('hdfs://path/to/file')")
    print("  counts = text_file.flatMap(lambda line: line.split()) \")
    print("                     .map(lambda word: (word, 1)) \")
    print("                     .reduceByKey(lambda a, b: a + b)")
    print("  counts.saveAsTextFile('hdfs://path/to/output')")
    print("- 数据处理:")
    print("  from pyspark.sql import SparkSession")
    print("  spark = SparkSession.builder.appName('DataProcessing').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)")
    print("  df.filter(df['age'] > 30).groupBy('department').count().show()")

# 9. Spark 性能优化
def spark_performance_optimization():
    print("\n9. Spark 性能优化:")
    print("- 数据缓存:")
    print("  rdd.cache() 或 rdd.persist(StorageLevel.MEMORY_AND_DISK)")
    print("- 数据分区:")
    print("  rdd.repartition(numPartitions)")
    print("- 广播变量:")
    print("  broadcast_var = sc.broadcast(large_data)")
    print("- 累加器:")
    print("  accumulator = sc.accumulator(0)")
    print("- 避免 shuffle:")
    print("  使用 reduceByKey 而非 groupByKey")
    print("- 合理设置资源:")
    print("  根据数据量调整 executor 内存和核心数")

# 10. Spark 监控
def spark_monitoring():
    print("\n10. Spark 监控:")
    print("- Web UI:")
    print("  - Driver Web UI: http://driver:4040")
    print("  - History Server: http://history-server:18080")
    print("- 日志:")
    print("  - Driver 日志: 标准输出或指定的日志文件")
    print("  - Executor 日志: 通过 YARN 或集群管理器查看")
    print("- 指标:")
    print("  - 内置指标系统")
    print("  - 集成 Prometheus 和 Grafana")

if __name__ == "__main__":
    # 执行所有演示
    spark_basics()
    spark_architecture()
    spark_installation()
    spark_running_modes()
    spark_core_components()
    spark_application_submission()
    spark_interactive_environments()
    spark_example_applications()
    spark_performance_optimization()
    spark_monitoring()
    
    print("\n" + "=" * 50)
    print("演示完成！")