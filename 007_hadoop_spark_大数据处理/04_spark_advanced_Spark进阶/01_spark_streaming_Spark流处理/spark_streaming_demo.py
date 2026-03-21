#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark Streaming 流处理演示

本脚本演示 Spark Streaming 的基本概念、操作和使用方法。
"""

import os
import sys

print("Spark Streaming 流处理演示")
print("=" * 50)

# 1. Spark Streaming 基本概念
def spark_streaming_basics():
    print("\n1. Spark Streaming 基本概念:")
    print("- Spark Streaming 是 Spark 用于处理实时流数据的模块")
    print("- 支持从多种数据源接收数据，如 Kafka、Flume、TCP 等")
    print("- 将流数据分割成小批次进行处理")
    print("- 提供与批处理类似的 API")
    print("- 支持状态管理和窗口操作")

# 2. Spark Streaming 架构
def spark_streaming_architecture():
    print("\n2. Spark Streaming 架构:")
    print("- Input DStreams: 输入流，从数据源接收数据")
    print("- DStreams: 离散流，是 RDD 的序列")
    print("- Transformations: 转换操作，如 map、filter、reduceByKey 等")
    print("- Output Operations: 输出操作，如 print、saveAsTextFiles 等")
    print("- Batch Interval: 批处理间隔，决定了处理数据的频率")

# 3. Spark Streaming 初始化
def spark_streaming_initialization():
    print("\n3. Spark Streaming 初始化:")
    print("- 创建 StreamingContext:")
    print("  from pyspark import SparkContext")
    print("  from pyspark.streaming import StreamingContext")
    print("  sc = SparkContext('local[*]', 'SparkStreamingDemo')")
    print("  ssc = StreamingContext(sc, 1)  # 1 秒批处理间隔")
    print("- 配置检查点:")
    print("  ssc.checkpoint('hdfs://path/to/checkpoint')")

# 4. 数据源连接
def data_source_connection():
    print("\n4. 数据源连接:")
    print("- TCP 数据源:")
    print("  lines = ssc.socketTextStream('localhost', 9999)")
    print("- Kafka 数据源:")
    print("  from pyspark.streaming.kafka import KafkaUtils")
    print("  kafkaParams = {}")
    print("  kafkaParams['bootstrap.servers'] = 'localhost:9092'")
    print("  kafkaParams['group.id'] = 'spark-streaming-group'")
    print("  topics = ['test-topic']")
    print("  kafkaStream = KafkaUtils.createDirectStream(ssc, topics, kafkaParams)")
    print("- Flume 数据源:")
    print("  from pyspark.streaming.flume import FlumeUtils")
    print("  flumeStream = FlumeUtils.createStream(ssc, 'localhost', 9999)")

# 5. DStream 操作
def dstream_operations():
    print("\n5. DStream 操作:")
    print("- 转换操作:")
    print("  # 映射操作")
    print("  mapped = lines.map(lambda x: x.upper())")
    print("  # 过滤操作")
    print("  filtered = lines.filter(lambda x: 'error' in x)")
    print("  # 扁平化操作")
    print("  words = lines.flatMap(lambda line: line.split())")
    print("  # 键值对操作")
    print("  pairs = words.map(lambda word: (word, 1))")
    print("  wordCounts = pairs.reduceByKey(lambda x, y: x + y)")
    print("- 输出操作:")
    print("  # 打印前 10 个元素")
    print("  wordCounts.pprint()")
    print("  # 保存为文本文件")
    print("  wordCounts.saveAsTextFiles('hdfs://path/to/output/prefix')")
    print("  # 保存到 HBase")
    print("  wordCounts.foreachRDD(lambda rdd: saveToHBase(rdd))")

# 6. 窗口操作
def window_operations():
    print("\n6. 窗口操作:")
    print("- 窗口转换:")
    print("  # 窗口大小为 10 秒，滑动间隔为 5 秒")
    print("  windowedWordCounts = pairs.reduceByKeyAndWindow(")
    print("      lambda x, y: x + y,  # 窗口内聚合")
    print("      lambda x, y: x - y,  # 窗口外聚合")
    print("      windowDuration=10,  # 窗口大小")
    print("      slideDuration=5  # 滑动间隔")
    print("  )")
    print("- 其他窗口操作:")
    print("  # 窗口内计数")
    print("  windowedCounts = lines.countByWindow(10, 5)")
    print("  # 窗口内最大值")
    print("  windowedMax = pairs.reduceByKeyAndWindow(lambda x, y: max(x, y), 10, 5)")

# 7. 状态管理
def stateful_operations():
    print("\n7. 状态管理:")
    print("- 有状态转换:")
    print("  # 定义更新函数")
    print("  def updateFunction(newValues, runningCount):")
    print("      if runningCount is None:")
    print("          runningCount = 0")
    print("      return sum(newValues) + runningCount if newValues else runningCount")
    print("  # 使用 updateStateByKey")
    print("  runningCounts = pairs.updateStateByKey(updateFunction)")
    print("- 检查点:")
    print("  # 设置检查点目录")
    print("  ssc.checkpoint('hdfs://path/to/checkpoint')")

# 8. 启动和停止
def start_stop():
    print("\n8. 启动和停止:")
    print("- 启动 StreamingContext:")
    print("  ssc.start()")
    print("  ssc.awaitTermination()")
    print("- 停止 StreamingContext:")
    print("  ssc.stop(stopSparkContext=True, stopGraceFully=True)")

# 9. 示例应用
def example_applications():
    print("\n9. 示例应用:")
    print("- 单词计数:")
    print("  from pyspark import SparkContext")
    print("  from pyspark.streaming import StreamingContext")
    print("  sc = SparkContext('local[*]', 'WordCount')")
    print("  ssc = StreamingContext(sc, 1)")
    print("  ssc.checkpoint('checkpoint')")
    print("  lines = ssc.socketTextStream('localhost', 9999)")
    print("  words = lines.flatMap(lambda line: line.split())")
    print("  pairs = words.map(lambda word: (word, 1))")
    print("  wordCounts = pairs.reduceByKey(lambda x, y: x + y)")
    print("  wordCounts.pprint()")
    print("  ssc.start()")
    print("  ssc.awaitTermination()")
    print("- 状态化单词计数:")
    print("  from pyspark import SparkContext")
    print("  from pyspark.streaming import StreamingContext")
    print("  sc = SparkContext('local[*]', 'StatefulWordCount')")
    print("  ssc = StreamingContext(sc, 1)")
    print("  ssc.checkpoint('checkpoint')")
    print("  lines = ssc.socketTextStream('localhost', 9999)")
    print("  words = lines.flatMap(lambda line: line.split())")
    print("  pairs = words.map(lambda word: (word, 1))")
    print("  def updateFunction(newValues, runningCount):")
    print("      if runningCount is None:")
    print("          runningCount = 0")
    print("      return sum(newValues) + runningCount if newValues else runningCount")
    print("  runningCounts = pairs.updateStateByKey(updateFunction)")
    print("  runningCounts.pprint()")
    print("  ssc.start()")
    print("  ssc.awaitTermination()")

# 10. 最佳实践
def best_practices():
    print("\n10. 最佳实践:")
    print("- 合理设置批处理间隔:")
    print("  根据数据速率和处理能力设置合适的 batch interval")
    print("- 使用检查点:")
    print("  启用检查点以支持状态恢复")
    print("- 优化资源配置:")
    print("  根据数据量调整 executor 内存和核心数")
    print("- 避免使用 collect():")
    print("  不要在流处理中使用 collect()，会导致内存溢出")
    print("- 合理使用窗口操作:")
    print("  窗口大小和滑动间隔要根据实际需求设置")
    print("- 错误处理:")
    print("  实现容错机制，处理数据源连接失败等情况")
    print("- 监控:")
    print("  监控流处理延迟和吞吐量")

if __name__ == "__main__":
    # 执行所有演示
    spark_streaming_basics()
    spark_streaming_architecture()
    spark_streaming_initialization()
    data_source_connection()
    dstream_operations()
    window_operations()
    stateful_operations()
    start_stop()
    example_applications()
    best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")