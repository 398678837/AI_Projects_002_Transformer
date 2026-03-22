# DataFrame 数据框详细文档

## 1. DataFrame 基本概念

DataFrame 是 Spark SQL 中的分布式数据集合，类似于关系型数据库中的表，具有行和列的结构。它提供了结构化数据处理能力，支持 SQL 查询和 DataFrame API，比 RDD 提供了更高的性能和更简洁的 API。

### 1.1 核心特性
- **结构化**：具有明确的模式（schema），包含列名和数据类型
- **分布式**：数据分布在多个节点上，支持并行处理
- **高性能**：利用 Catalyst 优化器和 Tungsten 执行引擎
- **易用性**：提供简洁的 API，支持多种操作
- **兼容性**：支持多种数据源和输出格式
- **类型安全**：在编译时检查类型错误（在 Scala 中）

### 1.2 DataFrame 与 RDD 的对比
| 特性 | DataFrame | RDD |
|------|-----------|-----|
| 数据结构 | 结构化，有 schema | 非结构化，无 schema |
| 性能 | 更高（优化器和执行引擎） | 较低（需要手动优化） |
| API | 更简洁，支持 SQL | 更灵活，功能更底层 |
| 类型安全 | 是的（Scala） | 是的（类型参数） |
| 适用场景 | 结构化数据处理 | 复杂数据处理 |

### 1.3 适用场景
- **结构化数据处理**：处理具有明确模式的数据
- **SQL 查询**：使用 SQL 语句进行数据查询和分析
- **数据清洗**：处理和转换数据
- **数据分析**：统计分析和聚合操作
- **数据集成**：与多种数据源和存储系统集成

## 2. DataFrame 创建

### 2.1 从 RDD 创建
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('DataFrameCreation').getOrCreate()

# 创建 RDD
rdd = spark.sparkContext.parallelize([(1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35)])

# 从 RDD 创建 DataFrame
# 方法 1：指定列名
df = spark.createDataFrame(rdd, ['id', 'name', 'age'])

# 方法 2：使用 Row 对象
from pyspark.sql import Row
row_rdd = rdd.map(lambda x: Row(id=x[0], name=x[1], age=x[2]))
df = spark.createDataFrame(row_rdd)
```

### 2.2 从集合创建
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('DataFrameCreation').getOrCreate()

# 从列表创建
data = [(1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35)]
df = spark.createDataFrame(data, ['id', 'name', 'age'])

# 从字典列表创建
dict_data = [{'id': 1, 'name': 'Alice', 'age': 25}, {'id': 2, 'name': 'Bob', 'age': 30}]
df = spark.createDataFrame(dict_data)
```

### 2.3 从文件创建
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('DataFrameCreation').getOrCreate()

# 从 CSV 文件创建
df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)

# 从 JSON 文件创建
df = spark.read.json('hdfs://path/to/data.json')

# 从 Parquet 文件创建
df = spark.read.parquet('hdfs://path/to/data.parquet')

# 从 ORC 文件创建
df = spark.read.orc('hdfs://path/to/data.orc')

# 从 JDBC 数据源创建
df = spark.read.format('jdbc') \
    .option('url', 'jdbc:mysql://localhost:3306/db') \
    .option('dbtable', 'table') \
    .option('user', 'username') \
    .option('password', 'password') \
    .load()
```

## 3. DataFrame 操作

### 3.1 查看数据
```python
# 显示前 20 行
df.show()

# 显示前 5 行
df.show(5)

# 显示前 5 行，不截断
df.show(5, False)

# 获取前 5 行
df.head(5)

# 获取前 5 行
df.take(5)

# 获取第一行
df.first()

# 随机采样 10%
df.sample(0.1).show()
```

### 3.2 查看结构
```python
# 显示 schema
df.printSchema()

# 获取列名
df.columns

# 获取列类型
df.dtypes

# 获取数据统计信息
df.describe().show()

# 获取特定列的统计信息
df.describe('age', 'salary').show()
```

### 3.3 选择列
```python
# 选择单个列
df.select('name').show()

# 选择多个列
df.select('name', 'age').show()

# 使用列对象选择
df.select(df['name'], df['age']).show()

# 使用表达式选择
df.select(df['name'], df['age'] + 1).show()

# 使用 col 函数选择
from pyspark.sql.functions import col
df.select(col('name'), col('age')).show()
```

### 3.4 过滤数据
```python
# 使用 filter 过滤
df.filter(df['age'] > 30).show()

# 使用 where 过滤（与 filter 功能相同）
df.where(df['age'] > 30).show()

# 使用 SQL 表达式过滤
df.filter('age > 30').show()

# 多个条件过滤
df.filter((df['age'] > 30) & (df['name'] == 'Bob')).show()

# 使用 isin 过滤
df.filter(df['name'].isin(['Alice', 'Bob'])).show()
```

### 3.5 排序数据
```python
# 升序排序
df.orderBy(df['age']).show()

# 降序排序
df.orderBy(df['age'].desc()).show()

# 使用 sort 排序（与 orderBy 功能相同）
df.sort(df['age']).show()

# 多列排序
df.orderBy(df['department'], df['age'].desc()).show()
```

### 3.6 分组聚合
```python
# 简单分组计数
df.groupBy('department').count().show()

# 分组聚合
df.groupBy('department').agg({'salary': 'avg', 'age': 'max'}).show()

# 使用聚合函数
trom pyspark.sql.functions import avg, max, min, count
df.groupBy('department').agg(
    avg('salary').alias('avg_salary'),
    max('age').alias('max_age'),
    min('age').alias('min_age'),
    count('*').alias('employee_count')
).show()
```

### 3.7 添加和修改列
```python
# 添加新列
df.withColumn('salary', df['age'] * 1000).show()

# 重命名列
df.withColumnRenamed('old_name', 'new_name').show()

# 修改列值
df.withColumn('age', df['age'] + 1).show()

# 条件添加列
from pyspark.sql.functions import when
df.withColumn('salary_level', 
              when(df['salary'] > 50000, 'high').otherwise('low')).show()
```

### 3.8 删除列
```python
# 删除单个列
df.drop('age').show()

# 删除多个列
df.drop('age', 'salary').show()

# 使用列表删除列
columns_to_drop = ['age', 'salary']
df.drop(*columns_to_drop).show()
```

## 4. DataFrame 转换

### 4.1 转换为 RDD
```python
# 转换为 RDD
rdd = df.rdd

# 查看 RDD 内容
print(rdd.collect())
```

### 4.2 转换为 Pandas DataFrame
```python
# 转换为 Pandas DataFrame
pandas_df = df.toPandas()

# 查看 Pandas DataFrame
print(pandas_df)
```

### 4.3 缓存 DataFrame
```python
# 缓存 DataFrame
df.cache()

# 持久化 DataFrame
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# 释放缓存
df.unpersist()
```

### 4.4 注册临时表
```python
# 注册临时表（会话级别）
df.createOrReplaceTempView('employees')

# 注册全局临时表（跨越多个会话）
df.createGlobalTempView('global_employees')
```

## 5. DataFrame SQL 操作

### 5.1 基本 SQL 查询
```python
# 注册临时表
df.createOrReplaceTempView('employees')

# 执行 SQL 查询
result = spark.sql('SELECT * FROM employees WHERE age > 30')
result.show()

# 复杂 SQL 查询
result = spark.sql('''
    SELECT department, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department
    HAVING avg_salary > 50000
    ORDER BY avg_salary DESC
''')
result.show()
```

### 5.2 SQL 函数
```python
# 使用 SQL 函数
result = spark.sql('''
    SELECT 
        name,
        age,
        UPPER(name) as upper_name,
        DATE_ADD(current_date(), 7) as next_week
    FROM employees
''')
result.show()
```

### 5.3 子查询
```python
# 使用子查询
result = spark.sql('''
    SELECT *
    FROM (
        SELECT department, AVG(salary) as avg_salary
        FROM employees
        GROUP BY department
    ) t
    WHERE avg_salary > 50000
''')
result.show()
```

## 6. DataFrame 连接操作

### 6.1 内连接
```python
# 创建两个 DataFrame
employees = spark.createDataFrame([(1, 'Alice', 25, 1), (2, 'Bob', 30, 2), (3, 'Charlie', 35, 1)], ['id', 'name', 'age', 'dept_id'])
departments = spark.createDataFrame([(1, 'Engineering'), (2, 'Marketing')], ['id', 'dept_name'])

# 内连接
joined = employees.join(departments, employees['dept_id'] == departments['id'], 'inner')
joined.show()
```

### 6.2 左连接
```python
# 左连接
joined = employees.join(departments, employees['dept_id'] == departments['id'], 'left')
joined.show()
```

### 6.3 右连接
```python
# 右连接
joined = employees.join(departments, employees['dept_id'] == departments['id'], 'right')
joined.show()
```

### 6.4 全连接
```python
# 全连接
joined = employees.join(departments, employees['dept_id'] == departments['id'], 'outer')
joined.show()
```

### 6.5 交叉连接
```python
# 交叉连接
joined = employees.crossJoin(departments)
joined.show()
```

### 6.6 广播连接
```python
# 广播连接（适用于小表）
from pyspark.sql.functions import broadcast
joined = employees.join(broadcast(departments), employees['dept_id'] == departments['id'])
joined.show()
```

## 7. DataFrame 数据写入

### 7.1 写入 CSV
```python
# 写入 CSV
df.write.csv('hdfs://path/to/output', header=True)

# 覆盖模式
df.write.mode('overwrite').csv('hdfs://path/to/output', header=True)

# 追加模式
df.write.mode('append').csv('hdfs://path/to/output', header=True)

# 忽略模式（如果文件已存在则忽略）
df.write.mode('ignore').csv('hdfs://path/to/output', header=True)

# 错误模式（如果文件已存在则抛出错误）
df.write.mode('error').csv('hdfs://path/to/output', header=True)
```

### 7.2 写入 JSON
```python
# 写入 JSON
df.write.json('hdfs://path/to/output')

# 覆盖模式
df.write.mode('overwrite').json('hdfs://path/to/output')
```

### 7.3 写入 Parquet
```python
# 写入 Parquet
df.write.parquet('hdfs://path/to/output')

# 覆盖模式
df.write.mode('overwrite').parquet('hdfs://path/to/output')
```

### 7.4 写入 Hive 表
```python
# 写入 Hive 表
df.write.saveAsTable('database.table')

# 覆盖模式
df.write.mode('overwrite').saveAsTable('database.table')
```

### 7.5 写入 JDBC
```python
# 写入 JDBC
df.write.format('jdbc') \
    .option('url', 'jdbc:mysql://localhost:3306/db') \
    .option('dbtable', 'table') \
    .option('user', 'username') \
    .option('password', 'password') \
    .mode('overwrite') \
    .save()
```

## 8. DataFrame 性能优化

### 8.1 缓存 DataFrame
```python
# 缓存频繁使用的 DataFrame
df.cache()

# 持久化到磁盘
df.persist(StorageLevel.DISK_ONLY)
```

### 8.2 分区优化
```python
# 重新分区（增加或减少分区数）
df.repartition(10)

# 减少分区数（避免 shuffle）
df.coalesce(2)

# 根据列分区
df.repartition('department')
```

### 8.3 谓词下推
```python
# 启用谓词下推
spark.conf.set('spark.sql.pushDownPredicate', 'true')
```

### 8.4 广播连接
```python
# 广播小表以减少 shuffle
from pyspark.sql.functions import broadcast
joined = employees.join(broadcast(departments), employees['dept_id'] == departments['id'])
```

### 8.5 避免使用 UDF
```python
# 避免使用 UDF，使用内置函数
# 不好的做法
from pyspark.sql.functions import udf
def square(x):
    return x * x
square_udf = udf(square)
df.withColumn('square_age', square_udf(df['age'])).show()

# 好的做法
from pyspark.sql.functions import col
df.withColumn('square_age', col('age') * col('age')).show()
```

### 8.6 使用列式存储
```python
# 使用 Parquet 格式存储数据
df.write.parquet('hdfs://path/to/output')

# 读取 Parquet 文件
df = spark.read.parquet('hdfs://path/to/output')
```

### 8.7 优化 Join 操作
```python
# 选择合适的连接类型
# 小表 join 大表：使用广播连接
# 大表 join 大表：使用 shuffle join

# 优化连接键
# 确保连接键的数据类型一致
# 避免使用复杂表达式作为连接键
```

### 8.8 合理设置 Spark 配置
```python
# 设置执行器内存
spark.conf.set('spark.executor.memory', '8g')

# 设置执行器核心数
spark.conf.set('spark.executor.cores', '4')

# 设置 shuffle 分区数
spark.conf.set('spark.sql.shuffle.partitions', '200')

# 设置默认并行度
spark.conf.set('spark.default.parallelism', '200')
```

## 9. DataFrame 示例应用

### 9.1 数据清洗
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace

# 创建 SparkSession
spark = SparkSession.builder.appName('DataCleaning').getOrCreate()

# 读取数据
df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)

# 查看数据
df.show()

# 处理缺失值
# 填充数值型列
filled_df = df.fillna({'age': 0, 'salary': 0})

# 填充字符串型列
filled_df = filled_df.fillna({'name': 'Unknown', 'department': 'Other'})

# 处理异常值
# 过滤年龄小于 0 的记录
filtered_df = filled_df.filter(col('age') >= 0)

# 处理薪资异常值
filtered_df = filtered_df.filter(col('salary') <= 1000000)

# 数据转换
# 标准化薪资
normalized_df = filtered_df.withColumn('normalized_salary', col('salary') / 1000)

# 添加薪资级别
final_df = normalized_df.withColumn('salary_level', 
                                  when(col('salary') > 50000, 'high') \
                                  .when(col('salary') > 30000, 'medium') \
                                  .otherwise('low'))

# 查看结果
final_df.show()

# 保存结果
final_df.write.parquet('hdfs://path/to/cleaned_data')

# 关闭 SparkSession
spark.stop()
```

### 9.2 数据分析
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, count, sum

# 创建 SparkSession
spark = SparkSession.builder.appName('DataAnalysis').getOrCreate()

# 读取销售数据
df = spark.read.csv('hdfs://path/to/sales.csv', header=True, inferSchema=True)

# 查看数据
df.show()

# 按产品类别分析销售数据
product_analysis = df.groupBy('product_category').agg(
    avg('sales').alias('avg_sales'),
    max('sales').alias('max_sales'),
    min('sales').alias('min_sales'),
    sum('sales').alias('total_sales'),
    count('*').alias('transaction_count')
)

# 按平均销售额排序
product_analysis = product_analysis.orderBy('avg_sales', ascending=False)

# 查看结果
product_analysis.show()

# 按地区分析销售数据
region_analysis = df.groupBy('region').agg(
    sum('sales').alias('total_sales'),
    count('*').alias('transaction_count')
)

# 查看结果
region_analysis.show()

# 按月份分析销售趋势
from pyspark.sql.functions import month, year

monthly_analysis = df.withColumn('year', year('date')) \
                    .withColumn('month', month('date')) \
                    .groupBy('year', 'month').agg(
                        sum('sales').alias('total_sales')
                    ) \
                    .orderBy('year', 'month')

# 查看结果
monthly_analysis.show()

# 保存分析结果
product_analysis.write.parquet('hdfs://path/to/product_analysis')
region_analysis.write.parquet('hdfs://path/to/region_analysis')
monthly_analysis.write.parquet('hdfs://path/to/monthly_analysis')

# 关闭 SparkSession
spark.stop()
```

### 9.3 数据集成
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('DataIntegration').getOrCreate()

# 读取员工数据
employees_df = spark.read.csv('hdfs://path/to/employees.csv', header=True, inferSchema=True)

# 读取部门数据
departments_df = spark.read.csv('hdfs://path/to/departments.csv', header=True, inferSchema=True)

# 读取薪资数据
salary_df = spark.read.csv('hdfs://path/to/salary.csv', header=True, inferSchema=True)

# 连接数据
# 员工和部门连接
emp_dept_df = employees_df.join(departments_df, employees_df['dept_id'] == departments_df['id'], 'inner')

# 添加薪资数据
final_df = emp_dept_df.join(salary_df, emp_dept_df['id'] == salary_df['emp_id'], 'inner')

# 选择需要的列
final_df = final_df.select(
    employees_df['id'].alias('employee_id'),
    employees_df['name'],
    employees_df['age'],
    departments_df['name'].alias('department'),
    salary_df['salary'],
    salary_df['bonus']
)

# 计算总收入
final_df = final_df.withColumn('total_income', final_df['salary'] + final_df['bonus'])

# 查看结果
final_df.show()

# 保存集成后的数据
final_df.write.parquet('hdfs://path/to/integrated_data')

# 关闭 SparkSession
spark.stop()
```

## 10. DataFrame 最佳实践

### 10.1 使用 Parquet 格式存储数据
- **优点**：列式存储，压缩率高，查询性能好
- **使用场景**：长期存储的数据，需要频繁查询的数据

### 10.2 合理设置分区数
- **原则**：分区数应根据集群资源和数据量调整
- **建议**：通常设置为集群核心数的 2-4 倍
- **注意**：分区数过多会增加调度开销，过少会降低并行度

### 10.3 使用广播变量处理小表连接
- **适用场景**：小表 join 大表
- **优点**：减少 shuffle，提高连接性能
- **使用方法**：`from pyspark.sql.functions import broadcast`

### 10.4 避免使用 collect() 处理大规模数据
- **问题**：collect() 会将所有数据拉取到驱动程序，可能导致内存溢出
- **解决方案**：使用 take()、first() 或 saveAs...() 方法

### 10.5 使用 DataFrame API 替代 RDD API
- **优点**：DataFrame API 更高级，性能更好，代码更简洁
- **适用场景**：结构化数据处理

### 10.6 合理使用缓存
- **适用场景**：数据需要多次使用
- **注意**：缓存会占用内存，使用完毕后应释放
- **方法**：`df.cache()` 或 `df.persist()`

### 10.7 使用内置函数替代 UDF
- **优点**：内置函数经过优化，性能更好
- **适用场景**：大多数数据转换操作
- **例外**：复杂的业务逻辑可能需要 UDF

### 10.8 优化 SQL 查询
- **原则**：避免全表扫描，使用索引，优化 join 操作
- **方法**：使用 WHERE 子句过滤数据，使用合适的连接类型

### 10.9 合理设置 Spark 配置
- **内存配置**：根据数据量和集群资源调整 executor 内存
- **并行度**：根据集群核心数调整默认并行度和 shuffle 分区数
- **其他配置**：根据具体场景调整其他配置参数

### 10.10 监控和调优
- **监控**：使用 Spark Web UI 监控应用执行情况
- **调优**：根据监控结果调整配置和代码
- **测试**：在生产环境前进行充分的测试

## 11. 总结

DataFrame 是 Spark SQL 中的核心数据结构，提供了强大的结构化数据处理能力。通过本文档的学习，您应该对 DataFrame 的基本概念、创建方法、操作类型、转换、SQL 操作、连接操作、数据写入、性能优化、示例应用和最佳实践有了全面的了解。

DataFrame 的设计理念和操作方式体现了 Spark 的核心价值：高性能、易用性和灵活性。通过合理使用 DataFrame 的各种特性，可以构建高效、可靠的分布式数据处理应用。

随着 Spark 的发展，DataFrame API 不断完善，成为了 Spark 中处理结构化数据的首选方式。掌握 DataFrame 的使用技巧，对于深入理解 Spark 和构建高性能的大数据应用至关重要。