#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataFrame 数据框演示

本脚本演示 Spark DataFrame 的基本概念、操作和使用方法。
"""

import os
import sys

print("DataFrame 数据框演示")
print("=" * 50)

# 1. DataFrame 基本概念
def dataframe_basics():
    print("\n1. DataFrame 基本概念:")
    print("- DataFrame 是 Spark SQL 中的分布式数据集合")
    print("- 类似于关系型数据库中的表，具有行和列")
    print("- 提供了结构化数据处理能力")
    print("- 支持 SQL 查询和 DataFrame API")
    print("- 比 RDD 提供了更高的性能和更简洁的 API")

# 2. DataFrame 创建
def dataframe_creation():
    print("\n2. DataFrame 创建:")
    print("- 从 RDD 创建:")
    print("  from pyspark.sql import SparkSession")
    print("  spark = SparkSession.builder.appName('DataFrameCreation').getOrCreate()")
    print("  rdd = spark.sparkContext.parallelize([(1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35)])")
    print("  df = spark.createDataFrame(rdd, ['id', 'name', 'age'])")
    print("- 从集合创建:")
    print("  data = [(1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35)]")
    print("  df = spark.createDataFrame(data, ['id', 'name', 'age'])")
    print("- 从文件创建:")
    print("  df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)")
    print("  df = spark.read.json('hdfs://path/to/data.json')")
    print("  df = spark.read.parquet('hdfs://path/to/data.parquet')")

# 3. DataFrame 操作
def dataframe_operations():
    print("\n3. DataFrame 操作:")
    print("- 查看数据:")
    print("  df.show()  # 显示前 20 行")
    print("  df.head(5)  # 获取前 5 行")
    print("  df.take(5)  # 获取前 5 行")
    print("  df.sample(0.1).show()  # 随机采样 10%")
    print("- 查看结构:")
    print("  df.printSchema()  # 显示 schema")
    print("  df.columns  # 获取列名")
    print("  df.dtypes  # 获取列类型")
    print("- 选择列:")
    print("  df.select('name', 'age').show()")
    print("  df.select(df['name'], df['age']).show()")
    print("- 过滤数据:")
    print("  df.filter(df['age'] > 30).show()")
    print("  df.where(df['age'] > 30).show()")
    print("- 排序数据:")
    print("  df.orderBy(df['age'].desc()).show()")
    print("  df.sort(df['age']).show()")
    print("- 分组聚合:")
    print("  df.groupBy('department').count().show()")
    print("  df.groupBy('department').agg({'salary': 'avg', 'age': 'max'}).show()")
    print("- 添加列:")
    print("  df.withColumn('salary', df['age'] * 1000).show()")
    print("  df.withColumnRenamed('old_name', 'new_name').show()")
    print("- 删除列:")
    print("  df.drop('age').show()")

# 4. DataFrame 转换
def dataframe_transformations():
    print("\n4. DataFrame 转换:")
    print("- 转换为 RDD:")
    print("  rdd = df.rdd")
    print("- 转换为 Pandas DataFrame:")
    print("  pandas_df = df.toPandas()")
    print("- 缓存 DataFrame:")
    print("  df.cache()")
    print("- 注册临时表:")
    print("  df.createOrReplaceTempView('employees')")

# 5. DataFrame SQL 操作
def dataframe_sql_operations():
    print("\n5. DataFrame SQL 操作:")
    print("- 注册临时表:")
    print("  df.createOrReplaceTempView('employees')")
    print("- 执行 SQL 查询:")
    print("  result = spark.sql('SELECT * FROM employees WHERE age > 30')")
    print("  result.show()")
    print("- 复杂 SQL 查询:")
    print("  result = spark.sql('''")
    print("      SELECT department, AVG(salary) as avg_salary")
    print("      FROM employees")
    print("      GROUP BY department")
    print("      HAVING avg_salary > 50000")
    print("      ORDER BY avg_salary DESC")
    print("  ''')")
    print("  result.show()")

# 6. DataFrame 连接操作
def dataframe_join_operations():
    print("\n6. DataFrame 连接操作:")
    print("- 创建两个 DataFrame:")
    print("  employees = spark.createDataFrame([(1, 'Alice', 25, 1), (2, 'Bob', 30, 2), (3, 'Charlie', 35, 1)], ['id', 'name', 'age', 'dept_id'])")
    print("  departments = spark.createDataFrame([(1, 'Engineering'), (2, 'Marketing')], ['id', 'dept_name'])")
    print("- 内连接:")
    print("  joined = employees.join(departments, employees['dept_id'] == departments['id'], 'inner')")
    print("  joined.show()")
    print("- 左连接:")
    print("  joined = employees.join(departments, employees['dept_id'] == departments['id'], 'left')")
    print("  joined.show()")
    print("- 右连接:")
    print("  joined = employees.join(departments, employees['dept_id'] == departments['id'], 'right')")
    print("  joined.show()")
    print("- 全连接:")
    print("  joined = employees.join(departments, employees['dept_id'] == departments['id'], 'outer')")
    print("  joined.show()")

# 7. DataFrame 数据写入
def dataframe_writing():
    print("\n7. DataFrame 数据写入:")
    print("- 写入 CSV:")
    print("  df.write.csv('hdfs://path/to/output', header=True)")
    print("- 写入 JSON:")
    print("  df.write.json('hdfs://path/to/output')")
    print("- 写入 Parquet:")
    print("  df.write.parquet('hdfs://path/to/output')")
    print("- 写入 Hive 表:")
    print("  df.write.saveAsTable('database.table')")
    print("- 覆盖模式:")
    print("  df.write.mode('overwrite').csv('hdfs://path/to/output')")
    print("- 追加模式:")
    print("  df.write.mode('append').csv('hdfs://path/to/output')")

# 8. DataFrame 性能优化
def dataframe_performance_optimization():
    print("\n8. DataFrame 性能优化:")
    print("- 缓存 DataFrame:")
    print("  df.cache()")
    print("- 分区优化:")
    print("  df.repartition(10)")
    print("  df.coalesce(2)")
    print("- 谓词下推:")
    print("  spark.conf.set('spark.sql.pushDownPredicate', 'true')")
    print("- 广播连接:")
    print("  from pyspark.sql.functions import broadcast")
    print("  joined = employees.join(broadcast(departments), employees['dept_id'] == departments['id'])")
    print("- 避免使用 UDF:")
    print("  使用内置函数替代 UDF")
    print("- 合理使用持久化级别:")
    print("  df.persist(StorageLevel.MEMORY_AND_DISK)")

# 9. DataFrame 示例应用
def dataframe_example_applications():
    print("\n9. DataFrame 示例应用:")
    print("- 数据清洗:")
    print("  from pyspark.sql import SparkSession")
    print("  from pyspark.sql.functions import col, when")
    print("  spark = SparkSession.builder.appName('DataCleaning').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)")
    print("  # 处理缺失值")
    print("  df = df.fillna({'age': 0, 'salary': 0})")
    print("  # 处理异常值")
    print("  df = df.filter(col('age') > 0)")
    print("  # 添加新列")
    print("  df = df.withColumn('salary_level', when(col('salary') > 50000, 'high').otherwise('low'))")
    print("  df.show()")
    print("- 数据分析:")
    print("  from pyspark.sql import SparkSession")
    print("  from pyspark.sql.functions import avg, max, min, count")
    print("  spark = SparkSession.builder.appName('DataAnalysis').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/sales.csv', header=True, inferSchema=True)")
    print("  # 按产品类别分析销售数据")
    print("  result = df.groupBy('product_category').agg(")
    print("      avg('sales').alias('avg_sales'),")
    print("      max('sales').alias('max_sales'),")
    print("      min('sales').alias('min_sales'),")
    print("      count('*').alias('transaction_count')")
    print("  )")
    print("  result.orderBy('avg_sales', ascending=False).show()")

# 10. DataFrame 最佳实践
def dataframe_best_practices():
    print("\n10. DataFrame 最佳实践:")
    print("- 使用 Parquet 格式存储数据，提高性能")
    print("- 合理设置分区数，提高并行度")
    print("- 使用广播变量处理小表连接")
    print("- 避免使用 collect() 处理大规模数据")
    print("- 使用 DataFrame API 替代 RDD API，提高性能")
    print("- 合理使用缓存，避免重复计算")
    print("- 使用内置函数替代 UDF，提高性能")
    print("- 优化 SQL 查询，避免全表扫描")

if __name__ == "__main__":
    # 执行所有演示
    dataframe_basics()
    dataframe_creation()
    dataframe_operations()
    dataframe_transformations()
    dataframe_sql_operations()
    dataframe_join_operations()
    dataframe_writing()
    dataframe_performance_optimization()
    dataframe_example_applications()
    dataframe_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")