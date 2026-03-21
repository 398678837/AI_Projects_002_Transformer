#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDD 弹性分布式数据集演示

本脚本演示 Spark RDD 的基本概念、操作和使用方法。
"""

import os
import sys

print("RDD 弹性分布式数据集演示")
print("=" * 50)

# 1. RDD 基本概念
def rdd_basics():
    print("\n1. RDD 基本概念:")
    print("- RDD (Resilient Distributed Dataset) 是 Spark 的基本数据结构")
    print("- 弹性: 能够自动恢复数据分区")
    print("- 分布式: 数据分布在多个节点上")
    print("- 数据集: 包含多个元素的集合")
    print("- 不可变: 一旦创建，不能修改，只能通过转换创建新的 RDD")

# 2. RDD 创建
def rdd_creation():
    print("\n2. RDD 创建:")
    print("- 从集合创建:")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'RDDCreation')")
    print("  rdd = sc.parallelize([1, 2, 3, 4, 5])")
    print("- 从文件创建:")
    print("  rdd = sc.textFile('hdfs://path/to/file.txt')")
    print("  rdd = sc.wholeTextFiles('hdfs://path/to/directory')")
    print("- 从其他 RDD 转换:")
    print("  new_rdd = rdd.map(lambda x: x * 2)")

# 3. RDD 操作
def rdd_operations():
    print("\n3. RDD 操作:")
    print("- 转换操作 (Transformations):")
    print("  - map: 对每个元素应用函数")
    print("  rdd.map(lambda x: x * 2)")
    print("  - filter: 过滤元素")
    print("  rdd.filter(lambda x: x > 3)")
    print("  - flatMap: 对每个元素应用函数并扁平化结果")
    print("  rdd.flatMap(lambda x: x.split())")
    print("  - reduceByKey: 按键聚合")
    print("  rdd.reduceByKey(lambda a, b: a + b)")
    print("  - sortBy: 排序")
    print("  rdd.sortBy(lambda x: x)")
    print("- 动作操作 (Actions):")
    print("  - collect: 收集所有元素到驱动程序")
    print("  rdd.collect()")
    print("  - count: 计算元素数量")
    print("  rdd.count()")
    print("  - first: 获取第一个元素")
    print("  rdd.first()")
    print("  - take: 获取前 n 个元素")
    print("  rdd.take(5)")
    print("  - reduce: 聚合元素")
    print("  rdd.reduce(lambda a, b: a + b)")
    print("  - saveAsTextFile: 保存为文本文件")
    print("  rdd.saveAsTextFile('hdfs://path/to/output')")

# 4. RDD 持久化
def rdd_persistence():
    print("\n4. RDD 持久化:")
    print("- 缓存 RDD:")
    print("  rdd.cache()")
    print("- 持久化级别:")
    print("  from pyspark import StorageLevel")
    print("  rdd.persist(StorageLevel.MEMORY_ONLY)")
    print("  rdd.persist(StorageLevel.MEMORY_AND_DISK)")
    print("  rdd.persist(StorageLevel.DISK_ONLY)")
    print("- 移除持久化:")
    print("  rdd.unpersist()")

# 5. RDD 分区
def rdd_partitioning():
    print("\n5. RDD 分区:")
    print("- 查看分区数:")
    print("  rdd.getNumPartitions()")
    print("- 重新分区:")
    print("  rdd.repartition(10)  # 增加分区数")
    print("  rdd.coalesce(2)  # 减少分区数")
    print("- 自定义分区:")
    print("  def my_partitioner(key):")
    print("      return hash(key) % 10")
    print("  rdd.partitionBy(10, my_partitioner)")

# 6. RDD 广播变量
def rdd_broadcast_variables():
    print("\n6. RDD 广播变量:")
    print("- 创建广播变量:")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'BroadcastExample')")
    print("  large_data = {1: 'one', 2: 'two', 3: 'three'}")
    print("  broadcast_var = sc.broadcast(large_data)")
    print("- 使用广播变量:")
    print("  rdd = sc.parallelize([1, 2, 3])")
    print("  result = rdd.map(lambda x: broadcast_var.value[x])")
    print("  print(result.collect())")

# 7. RDD 累加器
def rdd_accumulators():
    print("\n7. RDD 累加器:")
    print("- 创建累加器:")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'AccumulatorExample')")
    print("  accumulator = sc.accumulator(0)")
    print("- 使用累加器:")
    print("  rdd = sc.parallelize([1, 2, 3, 4, 5])")
    print("  def add_to_accumulator(x):")
    print("      global accumulator")
    print("      accumulator.add(x)")
    print("  rdd.foreach(add_to_accumulator)")
    print("  print(accumulator.value)")

# 8. RDD 依赖关系
def rdd_dependencies():
    print("\n8. RDD 依赖关系:")
    print("- 窄依赖 (Narrow Dependency):")
    print("  - 每个父 RDD 分区最多被一个子 RDD 分区使用")
    print("  - 例如: map, filter, flatMap")
    print("- 宽依赖 (Wide Dependency):")
    print("  - 多个子 RDD 分区依赖于同一个父 RDD 分区")
    print("  - 例如: reduceByKey, groupByKey, sortByKey")
    print("- 查看依赖关系:")
    print("  rdd.toDebugString()")

# 9. RDD 示例应用
def rdd_example_applications():
    print("\n9. RDD 示例应用:")
    print("- 单词计数:")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'WordCount')")
    print("  text_file = sc.textFile('hdfs://path/to/file.txt')")
    print("  word_counts = text_file.flatMap(lambda line: line.split()) \")
    print("                        .map(lambda word: (word, 1)) \")
    print("                        .reduceByKey(lambda a, b: a + b)")
    print("  word_counts.saveAsTextFile('hdfs://path/to/output')")
    print("- 数据过滤和聚合:")
    print("  from pyspark import SparkContext")
    print("  sc = SparkContext('local', 'DataProcessing')")
    print("  data = sc.parallelize([(1, 'A', 100), (2, 'B', 200), (3, 'A', 300), (4, 'B', 400)])")
    print("  result = data.filter(lambda x: x[2] > 150) \")
    print("               .map(lambda x: (x[1], x[2])) \")
    print("               .reduceByKey(lambda a, b: a + b)")
    print("  print(result.collect())")

# 10. RDD 最佳实践
def rdd_best_practices():
    print("\n10. RDD 最佳实践:")
    print("- 避免使用 collect() 处理大规模数据")
    print("- 合理使用持久化减少重复计算")
    print("- 使用广播变量传递大型只读数据")
    print("- 使用累加器进行计数和求和")
    print("- 避免使用 groupByKey()，优先使用 reduceByKey()")
    print("- 合理设置分区数，提高并行度")
    print("- 使用 coalesce() 减少分区数，避免 shuffle")
    print("- 避免在 RDD 操作中使用 Python 全局变量")

if __name__ == "__main__":
    # 执行所有演示
    rdd_basics()
    rdd_creation()
    rdd_operations()
    rdd_persistence()
    rdd_partitioning()
    rdd_broadcast_variables()
    rdd_accumulators()
    rdd_dependencies()
    rdd_example_applications()
    rdd_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")