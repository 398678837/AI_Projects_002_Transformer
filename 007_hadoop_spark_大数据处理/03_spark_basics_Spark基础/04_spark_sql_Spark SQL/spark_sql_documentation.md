# Spark SQL 详细文档

## 1. Spark SQL 基本概念

Spark SQL 是 Spark 用于处理结构化数据的模块，提供了 SQL 查询和 DataFrame API，支持多种数据源，如 Hive、Parquet、JSON 等，可以与 Spark Core 无缝集成，并提供了优化的执行计划。

### 1.1 核心特性
- **统一数据访问**：使用相同的接口访问不同的数据源
- **SQL 支持**：支持标准 SQL 查询
- **DataFrame API**：提供面向对象的编程接口
- **优化执行**：使用 Catalyst 优化器生成高效的执行计划
- **集成 Hive**：与 Hive 无缝集成，支持 Hive SQL
- **可扩展性**：支持自定义数据源和函数

### 1.2 架构组成
- **Catalyst 优化器**：负责 SQL 查询的解析、优化和执行计划生成
- **Tungsten 执行引擎**：提供高效的内存管理和执行
- **数据源 API**：支持多种数据源的统一访问
- **连接器**：与外部系统的集成

### 1.3 适用场景
- **数据查询和分析**：使用 SQL 进行数据查询和分析
- **数据转换**：将数据从一种格式转换为另一种格式
- **数据集成**：集成来自不同数据源的数据
- **数据处理**：处理和转换结构化数据
- **报表生成**：基于数据生成报表

## 2. SparkSession 创建

SparkSession 是 Spark SQL 的入口点，用于创建 DataFrame、执行 SQL 查询和访问数据源。

### 2.1 基本创建
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder \
    .appName('SparkSQLDemo') \
    .master('local[*]') \
    .getOrCreate()
```

### 2.2 配置 Hive 支持
```python
from pyspark.sql import SparkSession

# 创建支持 Hive 的 SparkSession
spark = SparkSession.builder \
    .appName('SparkSQLDemo') \
    .master('local[*]') \
    .enableHiveSupport() \
    .getOrCreate()
```

### 2.3 配置 SparkSession
```python
from pyspark.sql import SparkSession

# 创建带有配置的 SparkSession
spark = SparkSession.builder \
    .appName('SparkSQLDemo') \
    .master('local[*]') \
    .config('spark.sql.shuffle.partitions', '200') \
    .config('spark.executor.memory', '8g') \
    .getOrCreate()
```

## 3. 执行 SQL 查询

### 3.1 注册 DataFrame 为临时表
```python
# 创建 DataFrame
df = spark.createDataFrame([(1, 'Alice', 25, 'Engineering'), (2, 'Bob', 30, 'Marketing')], 
                           ['id', 'name', 'age', 'department'])

# 注册为临时表
df.createOrReplaceTempView('employees')
```

### 3.2 基本查询
```python
# 执行基本查询
result = spark.sql('SELECT * FROM employees')
result.show()

# 执行条件查询
result = spark.sql('SELECT * FROM employees WHERE age > 28')
result.show()

# 执行排序查询
result = spark.sql('SELECT * FROM employees ORDER BY age DESC')
result.show()
```

### 3.3 聚合查询
```python
# 执行聚合查询
result = spark.sql('''
    SELECT department, AVG(age) as avg_age, COUNT(*) as employee_count
    FROM employees
    GROUP BY department
    ORDER BY employee_count DESC
''')
result.show()
```

### 3.4 连接查询
```python
# 创建部门 DataFrame
departments_df = spark.createDataFrame([(1, 'Engineering'), (2, 'Marketing')], 
                                      ['id', 'dept_name'])

# 注册为临时表
departments_df.createOrReplaceTempView('departments')

# 执行连接查询
result = spark.sql('''
    SELECT e.id, e.name, e.age, d.dept_name
    FROM employees e
    JOIN departments d ON e.department = d.dept_name
''')
result.show()
```

### 3.5 子查询
```python
# 执行子查询
result = spark.sql('''
    SELECT *
    FROM (
        SELECT department, AVG(age) as avg_age
        FROM employees
        GROUP BY department
    ) t
    WHERE avg_age > 25
''')
result.show()
```

### 3.6 复杂查询
```python
# 执行复杂查询
result = spark.sql('''
    SELECT 
        department,
        COUNT(*) as total_employees,
        AVG(age) as avg_age,
        MIN(age) as min_age,
        MAX(age) as max_age
    FROM employees
    GROUP BY department
    HAVING total_employees > 1
    ORDER BY avg_age DESC
''')
result.show()
```

## 4. 数据源操作

### 4.1 读取数据
```python
# 读取 CSV 文件
df = spark.read.csv('hdfs://path/to/data.csv', header=True, inferSchema=True)

# 读取 JSON 文件
df = spark.read.json('hdfs://path/to/data.json')

# 读取 Parquet 文件
df = spark.read.parquet('hdfs://path/to/data.parquet')

# 读取 ORC 文件
df = spark.read.orc('hdfs://path/to/data.orc')

# 读取 Hive 表
df = spark.sql('SELECT * FROM database.table')

# 读取 JDBC 数据
df = spark.read.format('jdbc') \
    .option('url', 'jdbc:mysql://localhost:3306/db') \
    .option('dbtable', 'table') \
    .option('user', 'username') \
    .option('password', 'password') \
    .load()
```

### 4.2 写入数据
```python
# 写入 CSV 文件
df.write.csv('hdfs://path/to/output', header=True)

# 写入 JSON 文件
df.write.json('hdfs://path/to/output')

# 写入 Parquet 文件
df.write.parquet('hdfs://path/to/output')

# 写入 ORC 文件
df.write.orc('hdfs://path/to/output')

# 写入 Hive 表
df.write.saveAsTable('database.table')

# 写入 JDBC 数据
df.write.format('jdbc') \
    .option('url', 'jdbc:mysql://localhost:3306/db') \
    .option('dbtable', 'table') \
    .option('user', 'username') \
    .option('password', 'password') \
    .save()
```

### 4.3 写入模式
```python
# 覆盖模式
df.write.mode('overwrite').csv('hdfs://path/to/output')

# 追加模式
df.write.mode('append').csv('hdfs://path/to/output')

# 忽略模式（如果文件已存在则忽略）
df.write.mode('ignore').csv('hdfs://path/to/output')

# 错误模式（如果文件已存在则抛出错误）
df.write.mode('error').csv('hdfs://path/to/output')
```

## 5. 内置函数

Spark SQL 提供了丰富的内置函数，用于数据处理和转换。

### 5.1 字符串函数
```python
from pyspark.sql.functions import col, concat, lit, upper, lower, length, trim

# 字符串连接
df.select(concat(col('first_name'), lit(' '), col('last_name')).alias('full_name')).show()

# 字符串转换
df.select(upper(col('name')).alias('upper_name')).show()
df.select(lower(col('name')).alias('lower_name')).show()

# 字符串长度
df.select(length(col('name')).alias('name_length')).show()

# 去除空格
df.select(trim(col('name')).alias('trimmed_name')).show()
```

### 5.2 数学函数
```python
from pyspark.sql.functions import col, abs, sqrt, round, ceil, floor

# 绝对值
df.select(abs(col('value')).alias('absolute_value')).show()

# 平方根
df.select(sqrt(col('value')).alias('square_root')).show()

# 四舍五入
df.select(round(col('value'), 2).alias('rounded_value')).show()

# 向上取整
df.select(ceil(col('value')).alias('ceiled_value')).show()

# 向下取整
df.select(floor(col('value')).alias('floored_value')).show()
```

### 5.3 聚合函数
```python
from pyspark.sql.functions import col, sum, avg, max, min, count, countDistinct

# 求和
df.groupBy('department').agg(sum('salary').alias('total_salary')).show()

# 平均值
df.groupBy('department').agg(avg('salary').alias('avg_salary')).show()

# 最大值
df.groupBy('department').agg(max('salary').alias('max_salary')).show()

# 最小值
df.groupBy('department').agg(min('salary').alias('min_salary')).show()

# 计数
df.groupBy('department').agg(count('*').alias('employee_count')).show()

# 去重计数
df.groupBy('department').agg(countDistinct('position').alias('position_count')).show()
```

### 5.4 条件函数
```python
from pyspark.sql.functions import col, when, case, lit

# 简单条件
df.select(
    col('name'),
    when(col('salary') > 50000, 'high').otherwise('low').alias('salary_level')
).show()

# 多条件
df.select(
    col('name'),
    col('salary'),
    when(col('salary') > 80000, 'senior') \
    .when(col('salary') > 50000, 'mid') \
    .otherwise('entry').alias('salary_level')
).show()

# CASE 语句
from pyspark.sql.functions import expr
df.select(
    col('name'),
    col('salary'),
    expr('CASE WHEN salary > 80000 THEN "senior" WHEN salary > 50000 THEN "mid" ELSE "entry" END').alias('salary_level')
).show()
```

### 5.5 日期函数
```python
from pyspark.sql.functions import col, current_date, current_timestamp, date_format, datediff, months_between

# 当前日期
df.select(current_date().alias('current_date')).show()

# 当前时间戳
df.select(current_timestamp().alias('current_timestamp')).show()

# 日期格式化
df.select(date_format(col('hire_date'), 'yyyy-MM-dd').alias('formatted_date')).show()

# 日期差
df.select(datediff(current_date(), col('hire_date')).alias('days_employed')).show()

# 月份差
df.select(months_between(current_date(), col('hire_date')).alias('months_employed')).show()
```

### 5.6 其他函数
```python
from pyspark.sql.functions import col, isnull, coalesce, rand, rank

# 空值检查
df.select(isnull(col('salary')).alias('is_salary_null')).show()

# 空值替换
df.select(coalesce(col('salary'), lit(0)).alias('salary_with_default')).show()

# 随机数
df.select(rand().alias('random_number')).show()

# 排名
df.select(rank().over(window_spec).alias('rank')).show()
```

## 6. 自定义函数 (UDF)

自定义函数允许用户定义自己的函数，扩展 Spark SQL 的功能。

### 6.1 创建 UDF
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType

# 创建字符串处理 UDF
def to_upper(s):
    return s.upper() if s else s

upper_udf = udf(to_upper, StringType())

# 创建数值处理 UDF
def square(x):
    return x * x if x else x

square_udf = udf(square, IntegerType())
```

### 6.2 使用 UDF
```python
# 在 DataFrame API 中使用 UDF
df.select(upper_udf(col('name')).alias('upper_name')).show()
df.select(square_udf(col('age')).alias('age_squared')).show()

# 注册 UDF 到 SQL
spark.udf.register('to_upper', to_upper, StringType())
spark.udf.register('square', square, IntegerType())

# 在 SQL 中使用 UDF
spark.sql('SELECT to_upper(name) as upper_name FROM employees').show()
spark.sql('SELECT square(age) as age_squared FROM employees').show()
```

### 6.3 性能考虑
- **UDF 性能**：Python UDF 性能较低，尽量使用内置函数
- **向量化 UDF**：在 Spark 2.3+ 中，可以使用 Pandas UDF 提高性能
- **类型注解**：明确指定 UDF 的返回类型，提高性能

## 7. 窗口函数

窗口函数允许在一组行上执行计算，而不减少结果集中的行数。

### 7.1 导入窗口函数
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, percent_rank, ntile
```

### 7.2 创建窗口
```python
# 创建窗口规范
window_spec = Window \
    .partitionBy('department')  # 按部门分区
    .orderBy(col('salary').desc())  # 按薪资降序排序
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)  # 窗口范围
```

### 7.3 使用窗口函数
```python
# 行号
df.withColumn('row_number', row_number().over(window_spec)).show()

# 排名（可能有并列）
df.withColumn('rank', rank().over(window_spec)).show()

# 密集排名（无并列间隙）
df.withColumn('dense_rank', dense_rank().over(window_spec)).show()

# 百分比排名
df.withColumn('percent_rank', percent_rank().over(window_spec)).show()

# 分位数
df.withColumn('ntile', ntile(4).over(window_spec)).show()
```

### 7.4 聚合窗口函数
```python
from pyspark.sql.functions import sum, avg

# 累计求和
df.withColumn('cumulative_salary', sum('salary').over(window_spec)).show()

# 移动平均值
window_spec = Window \
    .partitionBy('department') \
    .orderBy('hire_date') \
    .rowsBetween(-2, 0)  # 前两行到当前行
df.withColumn('moving_avg_salary', avg('salary').over(window_spec)).show()
```

### 7.5 在 SQL 中使用窗口函数
```python
# 注册 DataFrame 为临时表
df.createOrReplaceTempView('employees')

# 执行窗口函数查询
result = spark.sql('''
    SELECT 
        name, 
        department, 
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as row_number,
        RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank,
        DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dense_rank
    FROM employees
''')
result.show()
```

## 8. 视图和表

### 8.1 临时视图
临时视图仅在当前 SparkSession 中可见，会话结束后自动删除。
```python
# 创建临时视图
df.createOrReplaceTempView('temp_view')

# 查询临时视图
spark.sql('SELECT * FROM temp_view').show()
```

### 8.2 全局临时视图
全局临时视图在所有 SparkSession 中可见，存储在 `global_temp` 数据库中。
```python
# 创建全局临时视图
df.createGlobalTempView('global_temp_view')

# 查询全局临时视图
spark.sql('SELECT * FROM global_temp.global_temp_view').show()
```

### 8.3 永久表
永久表存储在 Hive 元数据中，持久存在。
```python
# 创建永久表
df.write.saveAsTable('permanent_table')

# 查询永久表
spark.sql('SELECT * FROM permanent_table').show()

# 删除永久表
spark.sql('DROP TABLE IF EXISTS permanent_table')
```

### 8.4 分区表
分区表可以提高查询性能，特别是对于大型数据集。
```python
# 创建分区表
spark.sql('''
    CREATE TABLE partitioned_table (
        id INT,
        name STRING,
        salary DOUBLE
    ) 
    PARTITIONED BY (department STRING)
''')

# 插入分区数据
df.write.partitionBy('department').saveAsTable('partitioned_table')

# 查询分区表
spark.sql('SELECT * FROM partitioned_table WHERE department = "Engineering"').show()
```

## 9. 性能优化

### 9.1 缓存表
缓存表可以提高查询性能，避免重复计算。
```python
# 缓存表
spark.sql('CACHE TABLE employees')

# 解除缓存
spark.sql('UNCACHE TABLE employees')

# 缓存 DataFrame
df.cache()
df.persist()
```

### 9.2 广播表
对于小表，使用广播可以减少 shuffle 操作，提高连接性能。
```python
# 在 SQL 中使用广播提示
spark.sql('''
    SELECT /*+ BROADCAST(d) */ e.*, d.dept_name 
    FROM employees e 
    JOIN departments d ON e.dept_id = d.id
''').show()

# 在 DataFrame API 中使用广播
from pyspark.sql.functions import broadcast
joined_df = employees_df.join(broadcast(departments_df), employees_df['dept_id'] == departments_df['id'])
```

### 9.3 谓词下推
谓词下推可以将过滤操作下推到数据源，减少数据读取量。
```python
# 启用谓词下推
spark.conf.set('spark.sql.pushDownPredicate', 'true')
```

### 9.4 执行计划
查看执行计划可以帮助识别性能瓶颈。
```python
# 查看 DataFrame 执行计划
df.explain()
df.explain(extended=True)  # 查看详细执行计划

# 查看 SQL 执行计划
spark.sql('EXPLAIN SELECT * FROM employees WHERE age > 30').show()
spark.sql('EXPLAIN EXTENDED SELECT * FROM employees WHERE age > 30').show()
```

### 9.5 其他优化技巧
- **使用列式存储**：如 Parquet、ORC
- **合理设置分区数**：根据数据量和集群资源调整
- **使用合适的连接类型**：小表 join 大表使用广播连接
- **避免使用 UDF**：尽量使用内置函数
- **优化 SQL 查询**：避免全表扫描，使用索引
- **合理设置 Spark 配置**：根据具体场景调整配置参数

## 10. 示例应用

### 10.1 数据聚合
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('DataAggregation').getOrCreate()

# 读取销售数据
df = spark.read.csv('hdfs://path/to/sales.csv', header=True, inferSchema=True)

# 注册为临时表
df.createOrReplaceTempView('sales')

# 按产品和月份聚合销售数据
result = spark.sql('''
    SELECT 
        product_id, 
        MONTH(sale_date) as month, 
        SUM(amount) as total_sales,
        AVG(amount) as avg_sale,
        COUNT(*) as transaction_count
    FROM sales
    GROUP BY product_id, MONTH(sale_date)
    ORDER BY product_id, month
''')

# 查看结果
result.show()

# 保存结果
result.write.parquet('hdfs://path/to/sales_aggregation')

# 关闭 SparkSession
spark.stop()
```

### 10.2 数据转换
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# 创建 SparkSession
spark = SparkSession.builder.appName('DataTransformation').getOrCreate()

# 读取员工数据
df = spark.read.csv('hdfs://path/to/employees.csv', header=True, inferSchema=True)

# 注册为临时表
df.createOrReplaceTempView('employees')

# 添加薪资等级列
result = spark.sql('''
    SELECT 
        id, 
        name, 
        salary,
        CASE
            WHEN salary < 30000 THEN 'Entry Level'
            WHEN salary >= 30000 AND salary < 60000 THEN 'Mid Level'
            ELSE 'Senior Level'
        END as salary_level
    FROM employees
''')

# 查看结果
result.show()

# 保存结果
result.write.parquet('hdfs://path/to/employees_with_level')

# 关闭 SparkSession
spark.stop()
```

### 10.3 数据集成
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

# 注册为临时表
employees_df.createOrReplaceTempView('employees')
departments_df.createOrReplaceTempView('departments')
salary_df.createOrReplaceTempView('salary')

# 集成数据
result = spark.sql('''
    SELECT 
        e.id as employee_id,
        e.name,
        e.age,
        d.name as department,
        s.salary,
        s.bonus,
        (s.salary + s.bonus) as total_compensation
    FROM employees e
    JOIN departments d ON e.dept_id = d.id
    JOIN salary s ON e.id = s.emp_id
''')

# 查看结果
result.show()

# 保存结果
result.write.parquet('hdfs://path/to/integrated_employee_data')

# 关闭 SparkSession
spark.stop()
```

### 10.4 高级分析
```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

# 创建 SparkSession
spark = SparkSession.builder.appName('AdvancedAnalysis').getOrCreate()

# 读取销售数据
df = spark.read.csv('hdfs://path/to/sales.csv', header=True, inferSchema=True)

# 注册为临时表
df.createOrReplaceTempView('sales')

# 分析每个产品的月度销售趋势
result = spark.sql('''
    SELECT 
        product_id,
        YEAR(sale_date) as year,
        MONTH(sale_date) as month,
        SUM(amount) as monthly_sales,
        LAG(SUM(amount)) OVER (PARTITION BY product_id ORDER BY YEAR(sale_date), MONTH(sale_date)) as previous_month_sales,
        (SUM(amount) - LAG(SUM(amount)) OVER (PARTITION BY product_id ORDER BY YEAR(sale_date), MONTH(sale_date))) / LAG(SUM(amount)) OVER (PARTITION BY product_id ORDER BY YEAR(sale_date), MONTH(sale_date)) * 100 as growth_rate
    FROM sales
    GROUP BY product_id, YEAR(sale_date), MONTH(sale_date)
    ORDER BY product_id, year, month
''')

# 查看结果
result.show()

# 保存结果
result.write.parquet('hdfs://path/to/sales_trend_analysis')

# 关闭 SparkSession
spark.stop()
```

## 11. 总结

Spark SQL 是 Spark 生态系统中的重要组成部分，提供了强大的结构化数据处理能力。通过本文档的学习，您应该对 Spark SQL 的基本概念、SparkSession 创建、SQL 查询、数据源操作、内置函数、自定义函数、窗口函数、视图和表、性能优化以及示例应用有了全面的了解。

Spark SQL 的设计理念和功能特性体现了 Spark 的核心价值：高性能、易用性和灵活性。通过合理使用 Spark SQL 的各种特性，可以构建高效、可靠的大数据处理应用。

随着 Spark 的发展，Spark SQL 也在不断演进，引入了更多的特性和改进，如 Pandas UDF、自适应查询执行等，以提供更强大、更高效的数据处理能力。

掌握 Spark SQL 的使用技巧，对于深入理解 Spark 和构建高性能的大数据应用至关重要。通过本文档的学习，您应该能够在实际应用中灵活运用 Spark SQL 处理和分析结构化数据。