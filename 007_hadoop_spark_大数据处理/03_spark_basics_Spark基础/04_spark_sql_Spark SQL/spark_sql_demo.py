#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spark SQL 演示

本脚本演示 Spark SQL 的基本概念、操作和使用方法。
"""

import os
import sys

print("Spark SQL 演示")
print("=" * 50)

# 1. Spark SQL 基本概念
def spark_sql_basics():
    print("\n1. Spark SQL 基本概念:")
    print("- Spark SQL 是 Spark 用于处理结构化数据的模块")
    print("- 提供了 SQL 查询和 DataFrame API")
    print("- 支持多种数据源，如 Hive、Parquet、JSON 等")
    print("- 可以与 Spark Core 无缝集成")
    print("- 提供了优化的执行计划")

# 2. SparkSession 创建
def spark_session_creation():
    print("\n2. SparkSession 创建:")
    print("- 基本创建:")
    print("  from pyspark.sql import SparkSession")
    print("  spark = SparkSession.builder ")
    print("      .appName('SparkSQLDemo') ")
    print("      .master('local[*]') ")
    print("      .getOrCreate()")
    print("- 配置 Hive 支持:")
    print("  spark = SparkSession.builder ")
    print("      .appName('SparkSQLDemo') ")
    print("      .master('local[*]') ")
    print("      .enableHiveSupport() ")
    print("      .getOrCreate()")

# 3. 执行 SQL 查询
def sql_queries():
    print("\n3. 执行 SQL 查询:")
    print("- 注册 DataFrame 为临时表:")
    print("  df.createOrReplaceTempView('employees')")
    print("- 执行基本查询:")
    print("  result = spark.sql('SELECT * FROM employees')")
    print("  result.show()")
    print("- 执行条件查询:")
    print("  result = spark.sql('SELECT * FROM employees WHERE age > 30')")
    print("  result.show()")
    print("- 执行聚合查询:")
    print("  result = spark.sql('''")
    print("      SELECT department, AVG(salary) as avg_salary")
    print("      FROM employees")
    print("      GROUP BY department")
    print("      ORDER BY avg_salary DESC")
    print("  ''')")
    print("  result.show()")
    print("- 执行连接查询:")
    print("  result = spark.sql('''")
    print("      SELECT e.name, d.dept_name, e.salary")
    print("      FROM employees e")
    print("      JOIN departments d ON e.dept_id = d.id")
    print("  ''')")
    print("  result.show()")

# 4. 数据源操作
def data_source_operations():
    print("\n4. 数据源操作:")
    print("- 读取 CSV 文件:")
    print("  df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)")
    print("- 读取 JSON 文件:")
    print("  df = spark.read.json('hdfs://path/to/data.json')")
    print("- 读取 Parquet 文件:")
    print("  df = spark.read.parquet('hdfs://path/to/data.parquet')")
    print("- 读取 Hive 表:")
    print("  df = spark.sql('SELECT * FROM database.table')")
    print("- 写入 CSV 文件:")
    print("  df.write.csv('hdfs://path/to/output', header=True)")
    print("- 写入 Parquet 文件:")
    print("  df.write.parquet('hdfs://path/to/output')")
    print("- 写入 Hive 表:")
    print("  df.write.saveAsTable('database.table')")

# 5. 内置函数
def built_in_functions():
    print("\n5. 内置函数:")
    print("- 导入函数:")
    print("  from pyspark.sql.functions import col, concat, lit, sum, avg, max, min, count, when, date_format")
    print("- 字符串函数:")
    print("  df.select(concat(col('first_name'), lit(' '), col('last_name')).alias('full_name')).show()")
    print("- 数学函数:")
    print("  df.select(col('salary') * 1.1).alias('new_salary').show()")
    print("- 聚合函数:")
    print("  df.groupBy('department').agg(avg('salary').alias('avg_salary')).show()")
    print("- 条件函数:")
    print("  df.select(col('name'), when(col('salary') > 50000, 'high').otherwise('low').alias('salary_level')).show()")
    print("- 日期函数:")
    print("  df.select(date_format(col('hire_date'), 'yyyy-MM-dd').alias('formatted_date')).show()")

# 6. 自定义函数 (UDF)
def user_defined_functions():
    print("\n6. 自定义函数 (UDF):")
    print("- 创建 UDF:")
    print("  from pyspark.sql.functions import udf")
    print("  from pyspark.sql.types import StringType")
    print("  def to_upper(s):")
    print("      return s.upper() if s else s")
    print("  upper_udf = udf(to_upper, StringType())")
    print("- 使用 UDF:")
    print("  df.select(upper_udf(col('name')).alias('upper_name')).show()")
    print("- 注册 UDF 到 SQL:")
    print("  spark.udf.register('to_upper', to_upper, StringType())")
    print("  spark.sql('SELECT to_upper(name) FROM employees').show()")

# 7. 窗口函数
def window_functions():
    print("\n7. 窗口函数:")
    print("- 导入窗口函数:")
    print("  from pyspark.sql.window import Window")
    print("  from pyspark.sql.functions import row_number, rank, dense_rank")
    print("- 创建窗口:")
    print("  window_spec = Window.partitionBy('department').orderBy(col('salary').desc())")
    print("- 使用窗口函数:")
    print("  df.withColumn('rank', rank().over(window_spec)) \")
    print("    .withColumn('row_number', row_number().over(window_spec)) \")
    print("    .show()")
    print("- 计算部门内工资排名:")
    print("  result = spark.sql('''")
    print("      SELECT name, department, salary,")
    print("             RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank")
    print("      FROM employees")
    print("  ''')")
    print("  result.show()")

# 8. 视图和表
def views_and_tables():
    print("\n8. 视图和表:")
    print("- 临时视图:")
    print("  df.createOrReplaceTempView('temp_view')")
    print("  spark.sql('SELECT * FROM temp_view').show()")
    print("- 全局临时视图:")
    print("  df.createGlobalTempView('global_temp_view')")
    print("  spark.sql('SELECT * FROM global_temp.global_temp_view').show()")
    print("- 永久表:")
    print("  df.write.saveAsTable('permanent_table')")
    print("  spark.sql('SELECT * FROM permanent_table').show()")

# 9. 性能优化
def performance_optimization():
    print("\n9. 性能优化:")
    print("- 缓存表:")
    print("  spark.sql('CACHE TABLE employees')")
    print("- 广播表:")
    print("  spark.sql('SELECT /*+ BROADCAST(d) */ e.*, d.dept_name FROM employees e JOIN departments d ON e.dept_id = d.id').show()")
    print("- 分区表:")
    print("  spark.sql('CREATE TABLE partitioned_table (id INT, name STRING, date STRING) PARTITIONED BY (date)')")
    print("- 谓词下推:")
    print("  spark.conf.set('spark.sql.pushDownPredicate', 'true')")
    print("- 执行计划:")
    print("  df.explain()")
    print("  spark.sql('EXPLAIN SELECT * FROM employees WHERE age > 30').show()")

# 10. 示例应用
def example_applications():
    print("\n10. 示例应用:")
    print("- 数据聚合:")
    print("  from pyspark.sql import SparkSession")
    print("  spark = SparkSession.builder.appName('DataAggregation').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/sales.csv', header=True, inferSchema=True)")
    print("  # 按产品和月份聚合销售数据")
    print("  result = spark.sql('''")
    print("      SELECT product_id, MONTH(sale_date) as month, SUM(amount) as total_sales")
    print("      FROM sales")
    print("      GROUP BY product_id, MONTH(sale_date)")
    print("      ORDER BY product_id, month")
    print("  ''')")
    print("  result.show()")
    print("- 数据转换:")
    print("  from pyspark.sql import SparkSession")
    print("  from pyspark.sql.functions import col, when")
    print("  spark = SparkSession.builder.appName('DataTransformation').getOrCreate()")
    print("  df = spark.read.csv('hdfs://path/to/employees.csv', header=True, inferSchema=True)")
    print("  # 添加薪资等级列")
    print("  df.createOrReplaceTempView('employees')")
    print("  result = spark.sql('''")
    print("      SELECT id, name, salary,")
    print("             CASE")
    print("                 WHEN salary < 30000 THEN 'Entry Level'")
    print("                 WHEN salary >= 30000 AND salary < 60000 THEN 'Mid Level'")
    print("                 ELSE 'Senior Level'")
    print("             END as salary_level")
    print("      FROM employees")
    print("  ''')")
    print("  result.show()")

if __name__ == "__main__":
    # 执行所有演示
    spark_sql_basics()
    spark_session_creation()
    sql_queries()
    data_source_operations()
    built_in_functions()
    user_defined_functions()
    window_functions()
    views_and_tables()
    performance_optimization()
    example_applications()
    
    print("\n" + "=" * 50)
    print("演示完成！")