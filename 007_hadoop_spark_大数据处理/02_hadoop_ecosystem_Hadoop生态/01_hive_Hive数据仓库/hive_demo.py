#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hive 数据仓库演示

本脚本演示 Hive 的基本概念、操作和使用方法。
"""

import os
import sys

print("Hive 数据仓库演示")
print("=" * 50)

# 1. Hive 基本概念
def hive_basics():
    print("\n1. Hive 基本概念:")
    print("- Hive 是基于 Hadoop 的数据仓库工具")
    print("- 提供类 SQL 语言 (HQL) 来查询和分析数据")
    print("- 将 SQL 语句转换为 MapReduce 作业执行")
    print("- 适合处理大规模数据集的批处理任务")

# 2. Hive 架构
def hive_architecture():
    print("\n2. Hive 架构:")
    print("- Hive CLI: 命令行界面")
    print("- HiveServer2: 提供 JDBC/ODBC 接口")
    print("- Metastore: 存储元数据信息")
    print("- Driver: 解析 HQL 语句")
    print("- Compiler: 编译 HQL 为执行计划")
    print("- Executor: 执行 MapReduce 作业")

# 3. Hive 数据类型
def hive_data_types():
    print("\n3. Hive 数据类型:")
    print("- 基本数据类型:")
    print("  - 整型: TINYINT, SMALLINT, INT, BIGINT")
    print("  - 浮点型: FLOAT, DOUBLE")
    print("  - 字符串: STRING, VARCHAR, CHAR")
    print("  - 布尔型: BOOLEAN")
    print("  - 日期时间: DATE, TIMESTAMP")
    print("- 复杂数据类型:")
    print("  - 数组: ARRAY<type>")
    print("  - 映射: MAP<key_type, value_type>")
    print("  - 结构: STRUCT<field1:type1, field2:type2, ...>")
    print("  - 联合: UNIONTYPE<type1, type2, ...>")

# 4. Hive 表操作
def hive_table_operations():
    print("\n4. Hive 表操作:")
    print("- 创建表:")
    print("  CREATE TABLE IF NOT EXISTS employees (")
    print("    id INT,")
    print("    name STRING,")
    print("    salary FLOAT,")
    print("    hire_date DATE")
    print("  ) ROW FORMAT DELIMITED")
    print("  FIELDS TERMINATED BY '\\t'")
    print("  STORED AS TEXTFILE;")
    print("- 查看表结构:")
    print("  DESCRIBE employees;")
    print("- 查看表列表:")
    print("  SHOW TABLES;")
    print("- 删除表:")
    print("  DROP TABLE IF EXISTS employees;")

# 5. Hive 数据操作
def hive_data_operations():
    print("\n5. Hive 数据操作:")
    print("- 加载数据:")
    print("  LOAD DATA LOCAL INPATH '/path/to/local/file' INTO TABLE employees;")
    print("  LOAD DATA INPATH '/path/to/hdfs/file' INTO TABLE employees;")
    print("- 插入数据:")
    print("  INSERT INTO TABLE employees VALUES (1, 'John', 50000.0, '2023-01-01');")
    print("  INSERT OVERWRITE TABLE employees SELECT * FROM old_employees;")
    print("- 导出数据:")
    print("  INSERT OVERWRITE LOCAL DIRECTORY '/path/to/local/dir' SELECT * FROM employees;")
    print("  INSERT OVERWRITE DIRECTORY '/path/to/hdfs/dir' SELECT * FROM employees;")

# 6. Hive 查询操作
def hive_query_operations():
    print("\n6. Hive 查询操作:")
    print("- 基本查询:")
    print("  SELECT * FROM employees;")
    print("  SELECT name, salary FROM employees WHERE salary > 50000;")
    print("- 聚合查询:")
    print("  SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department;")
    print("  SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department HAVING COUNT(*) > 10;")
    print("- 连接查询:")
    print("  SELECT e.name, d.department_name FROM employees e JOIN departments d ON e.department_id = d.id;")
    print("- 排序查询:")
    print("  SELECT * FROM employees ORDER BY salary DESC;")

# 7. Hive 分区和分桶
def hive_partitioning_bucketing():
    print("\n7. Hive 分区和分桶:")
    print("- 分区表:")
    print("  CREATE TABLE employees_partitioned (")
    print("    id INT,")
    print("    name STRING,")
    print("    salary FLOAT")
    print("  ) PARTITIONED BY (hire_year INT, hire_month INT)")
    print("  ROW FORMAT DELIMITED")
    print("  FIELDS TERMINATED BY '\\t';")
    print("  ALTER TABLE employees_partitioned ADD PARTITION (hire_year=2023, hire_month=1);")
    print("- 分桶表:")
    print("  CREATE TABLE employees_bucketed (")
    print("    id INT,")
    print("    name STRING,")
    print("    salary FLOAT")
    print("  ) CLUSTERED BY (id) INTO 4 BUCKETS")
    print("  ROW FORMAT DELIMITED")
    print("  FIELDS TERMINATED BY '\\t';")

# 8. Hive 函数
def hive_functions():
    print("\n8. Hive 函数:")
    print("- 内置函数:")
    print("  - 字符串函数: CONCAT, SUBSTR, LOWER, UPPER")
    print("  - 数学函数: SUM, AVG, MAX, MIN, COUNT")
    print("  - 日期函数: CURRENT_DATE, DATE_ADD, DATE_DIFF")
    print("  - 条件函数: IF, CASE, COALESCE")
    print("  - 聚合函数: GROUP_CONCAT, COLLECT_SET")
    print("- 自定义函数 (UDF):")
    print("  - 编写 Java 类实现 UDF 接口")
    print("  - 编译打包为 JAR 文件")
    print("  - 在 Hive 中添加 JAR 并创建函数")
    print("  CREATE FUNCTION my_function AS 'com.example.MyUDF' USING JAR 'hdfs://path/to/jar';")

# 9. Hive 性能优化
def hive_performance_optimization():
    print("\n9. Hive 性能优化:")
    print("- 表设计优化:")
    print("  - 使用分区表减少数据扫描")
    print("  - 使用分桶表提高查询效率")
    print("  - 选择合适的文件格式 (ORC, Parquet)")
    print("- 查询优化:")
    print("  - 谓词下推: WHERE 条件尽可能早地执行")
    print("  - 列裁剪: 只选择需要的列")
    print("  - 分区裁剪: 只查询相关分区")
    print("  - 使用 JOIN 优化: 小表在前，大表在后")
    print("- 配置优化:")
    print("  - 设置合适的 MapReduce 任务数")
    print("  - 启用压缩减少数据传输")
    print("  - 启用向量查询提高性能")

# 10. Hive 常见问题
def hive_common_issues():
    print("\n10. Hive 常见问题:")
    print("- 数据加载失败:")
    print("  - 检查文件格式是否正确")
    print("  - 检查权限是否正确")
    print("- 查询执行缓慢:")
    print("  - 检查是否使用了分区裁剪")
    print("  - 检查 JOIN 操作是否优化")
    print("  - 考虑使用更高效的文件格式")
    print("- 内存溢出:")
    print("  - 增加 MapReduce 任务内存")
    print("  - 减少单个任务处理的数据量")
    print("- 元数据丢失:")
    print("  - 定期备份 Metastore")
    print("  - 配置 Metastore 高可用")

if __name__ == "__main__":
    # 执行所有演示
    hive_basics()
    hive_architecture()
    hive_data_types()
    hive_table_operations()
    hive_data_operations()
    hive_query_operations()
    hive_partitioning_bucketing()
    hive_functions()
    hive_performance_optimization()
    hive_common_issues()
    
    print("\n" + "=" * 50)
    print("演示完成！")