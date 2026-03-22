# Structured Streaming 结构化流详细文档

## 1. Structured Streaming 基本概念

Structured Streaming 是 Spark 2.0 引入的流处理 API，基于 DataFrame API，提供了声明式的流处理能力。它将流数据视为无限增长的表，支持与批处理相同的操作，并提供了 exactly-once 语义保证。

### 1.1 核心特性
- **声明式 API**：使用与批处理相同的 DataFrame/Dataset API
- **无限表模型**：将流数据视为无限增长的表
- **容错保证**：提供 exactly-once 语义
- **低延迟**：支持毫秒级处理延迟
- **高吞吐量**：处理大规模流数据
- **状态管理**：支持有状态的流处理
- **集成性**：与 Spark SQL、MLlib 等无缝集成

### 1.2 适用场景
- **实时数据处理**：处理实时生成的数据
- **实时分析**：实时分析数据流
- **实时监控**：监控系统状态和指标
- **实时 ETL**：实时提取、转换和加载数据
- **实时机器学习**：实时训练和预测模型
- **事件处理**：处理和响应事件流

## 2. 流处理模型

### 2.1 核心概念
- **输入表**：从数据源接收的流数据，持续增长
- **查询**：对输入表的转换操作，与批处理查询语法相同
- **结果表**：查询结果，根据处理模式更新
- **输出**：将结果表输出到外部系统
- **触发器**：控制流处理的执行频率

### 2.2 处理模式
- **Append 模式**：只输出新添加的行，适用于结果表只增长不更新的场景
- **Complete 模式**：输出所有结果，适用于需要全量结果的场景
- **Update 模式**：只输出更新的行，适用于需要增量更新的场景

### 2.3 时间概念
- **事件时间**：事件发生的时间，通常包含在数据中
- **处理时间**：数据被处理的时间
- **摄取时间**：数据被摄入系统的时间

## 3. 数据源连接

### 3.1 内置数据源

#### 3.1.1 TCP 套接字
```python
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.appName('StructuredStreamingDemo').getOrCreate()

# 从 TCP 套接字读取数据
lines = spark.readStream.format('socket') \
    .option('host', 'localhost') \
    .option('port', 9999) \
    .load()
```

#### 3.1.2 Kafka
```python
# 从 Kafka 读取数据
df = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'test-topic') \
    .option('startingOffsets', 'earliest') \
    .load()

# 处理 Kafka 数据
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, IntegerType

schema = StructType([
    StructField('id', IntegerType(), True),
    StructField('name', StringType(), True),
    StructField('value', IntegerType(), True)
])

# 解析 JSON 数据
df = df.selectExpr('CAST(value AS STRING)') \
    .select(from_json(col('value'), schema).alias('data')) \
    .select('data.*')
```

#### 3.1.3 文件系统
```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# 定义 schema
schema = StructType([
    StructField('name', StringType(), True),
    StructField('age', IntegerType(), True),
    StructField('salary', IntegerType(), True)
])

# 从文件系统读取数据
df = spark.readStream.format('csv') \
    .option('header', 'true') \
    .schema(schema) \
    .load('hdfs://path/to/directory')
```

#### 3.1.4 Rate 数据源（用于测试）
```python
# 生成测试数据
df = spark.readStream.format('rate') \
    .option('rowsPerSecond', 10) \
    .option('rampUpTime', 5) \
    .option('numPartitions', 2) \
    .load()
```

### 3.2 自定义数据源
通过实现 `Source` 接口，可以创建自定义数据源。

## 4. 流数据处理

### 4.1 基本操作
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode

# 创建 SparkSession
spark = SparkSession.builder.appName('StructuredStreamingDemo').getOrCreate()

# 从 TCP 套接字读取数据
lines = spark.readStream.format('socket') \
    .option('host', 'localhost') \
    .option('port', 9999) \
    .load()

# 分词
words = lines.select(explode(split(lines['value'], ' ')).alias('word'))

# 计数
wordCounts = words.groupBy('word').count()
```

### 4.2 窗口操作
```python
from pyspark.sql.functions import window, current_timestamp

# 添加时间戳
linesWithTimestamp = lines.withColumn('timestamp', current_timestamp())

# 分词并保留时间戳
wordsWithTimestamp = linesWithTimestamp.select(
    explode(split(linesWithTimestamp['value'], ' ')).alias('word'),
    linesWithTimestamp['timestamp']
)

# 窗口计数（10分钟窗口，5分钟滑动）
windowedCounts = wordsWithTimestamp.groupBy(
    window(wordsWithTimestamp['timestamp'], '10 minutes', '5 minutes'),
    wordsWithTimestamp['word']
).count()
```

### 4.3 状态操作
```python
# 有状态的聚合
def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues) + runningCount if newValues else runningCount

# 使用 mapGroupsWithState（注意：Python API 中有限制）
from pyspark.sql.functions import col

# 按键分组并聚合
statefulCounts = words.groupBy('word').agg({'word': 'count'})
```

### 4.4 流-批连接
```python
# 加载静态数据（批处理）
staticDF = spark.read.parquet('hdfs://path/to/static/data')

# 流数据与静态数据连接
joinedDF = words.join(
    staticDF, 
    words['word'] == staticDF['word'], 
    'left'
)
```

## 5. 输出操作

### 5.1 输出模式
- **Append 模式**：只输出新添加的行
- **Complete 模式**：输出所有结果
- **Update 模式**：只输出更新的行

### 5.2 输出目标

#### 5.2.1 控制台
```python
# 输出到控制台
query = wordCounts.writeStream \
    .outputMode('complete') \
    .format('console') \
    .start()
```

#### 5.2.2 文件系统
```python
# 输出到文件系统
query = wordCounts.writeStream \
    .outputMode('append') \
    .format('parquet') \
    .option('path', 'hdfs://path/to/output') \
    .option('checkpointLocation', 'hdfs://path/to/checkpoint') \
    .start()
```

#### 5.2.3 Kafka
```python
# 输出到 Kafka
query = wordCounts.writeStream \
    .outputMode('update') \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('topic', 'output-topic') \
    .option('checkpointLocation', 'hdfs://path/to/checkpoint') \
    .start()
```

#### 5.2.4 内存表
```python
# 输出到内存表
query = wordCounts.writeStream \
    .outputMode('complete') \
    .format('memory') \
    .queryName('word_counts') \
    .start()

# 从内存表读取数据
spark.sql('SELECT * FROM word_counts').show()
```

#### 5.2.5 自定义输出
通过实现 `Sink` 接口，可以创建自定义输出目标。

## 6. 检查点

### 6.1 检查点的作用
- **保存流处理的状态**：包括处理进度、状态数据等
- **支持故障恢复**：当系统故障时，可从检查点恢复处理
- **确保 exactly-once 语义**：通过状态保存和恢复，保证处理的一致性

### 6.2 设置检查点
```python
# 设置检查点
query = wordCounts.writeStream \
    .option('checkpointLocation', 'hdfs://path/to/checkpoint') \
    .start()
```

### 6.3 检查点目录结构
检查点目录包含以下内容：
- **offsets**：存储数据源的偏移量
- **commits**：存储已提交的批处理
- **metadata**：存储流处理的元数据
- **state**：存储状态数据

## 7. 流处理控制

### 7.1 启动查询
```python
# 启动查询
query = wordCounts.writeStream \
    .outputMode('complete') \
    .format('console') \
    .start()
```

### 7.2 等待查询完成
```python
# 等待查询完成（阻塞）
query.awaitTermination()

# 等待指定时间
query.awaitTermination(10000)  # 10秒
```

### 7.3 停止查询
```python
# 停止查询
query.stop()
```

### 7.4 查看查询状态
```python
# 查看查询状态
print(query.status)
print(query.lastProgress)

# 查看所有活跃查询
for q in spark.streams.active:
    print(q.name)
    print(q.status)
```

### 7.5 触发器
```python
from pyspark.sql.streaming import Trigger

# 固定间隔触发器
query = wordCounts.writeStream \
    .trigger(Trigger.ProcessingTime('10 seconds')) \
    .start()

# 一次触发器（只处理一次）
query = wordCounts.writeStream \
    .trigger(Trigger.Once()) \
    .start()

# 连续触发器（低延迟）
query = wordCounts.writeStream \
    .trigger(Trigger.Continuous('1 second')) \
    .start()
```

## 8. 状态管理

### 8.1 有状态操作
- **分组聚合**：按键分组并聚合
- **窗口操作**：基于时间窗口的聚合
- **流-流连接**：两个流之间的连接

### 8.2 状态存储
- **内存存储**：默认存储在内存中
- **磁盘存储**：通过检查点持久化到磁盘

### 8.3 状态清理
```python
# 设置状态超时
query = wordCounts.writeStream \
    .outputMode('update') \
    .option('checkpointLocation', 'hdfs://path/to/checkpoint') \
    .option('spark.sql.streaming.stateStore.providerClass', 'org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider') \
    .start()
```

## 9. 示例应用

### 9.1 实时单词计数
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode

# 创建 SparkSession
spark = SparkSession.builder.appName('WordCount').getOrCreate()

# 从 TCP 套接字读取数据
lines = spark.readStream.format('socket') \
    .option('host', 'localhost') \
    .option('port', 9999) \
    .load()

# 分词
words = lines.select(explode(split(lines['value'], ' ')).alias('word'))

# 计数
wordCounts = words.groupBy('word').count()

# 输出到控制台
query = wordCounts.writeStream \
    .outputMode('complete') \
    .format('console') \
    .start()

# 等待查询完成
query.awaitTermination()
```

### 9.2 窗口单词计数
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, window, current_timestamp

# 创建 SparkSession
spark = SparkSession.builder.appName('WindowedWordCount').getOrCreate()

# 从 TCP 套接字读取数据
lines = spark.readStream.format('socket') \
    .option('host', 'localhost') \
    .option('port', 9999) \
    .load()

# 添加时间戳
linesWithTimestamp = lines.withColumn('timestamp', current_timestamp())

# 分词
words = linesWithTimestamp.select(
    explode(split(linesWithTimestamp['value'], ' ')).alias('word'),
    linesWithTimestamp['timestamp']
)

# 窗口计数（10分钟窗口，5分钟滑动）
windowedCounts = words.groupBy(
    window(words['timestamp'], '10 minutes', '5 minutes'),
    words['word']
).count()

# 输出到控制台
query = windowedCounts.writeStream \
    .outputMode('complete') \
    .format('console') \
    .start()

# 等待查询完成
query.awaitTermination()
```

### 9.3 Kafka 流处理
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import StructType, StringType, IntegerType

# 创建 SparkSession
spark = SparkSession.builder.appName('KafkaStreamProcessing').getOrCreate()

# 从 Kafka 读取数据
df = spark.readStream.format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('subscribe', 'input-topic') \
    .option('startingOffsets', 'earliest') \
    .load()

# 定义 schema
schema = StructType([
    StructField('id', IntegerType(), True),
    StructField('name', StringType(), True),
    StructField('value', IntegerType(), True)
])

# 解析 JSON 数据
df = df.selectExpr('CAST(value AS STRING)') \
    .select(from_json(col('value'), schema).alias('data')) \
    .select('data.*')

# 处理数据
processedDF = df.filter(df['value'] > 100) \
    .withColumn('processed_value', col('value') * 2)

# 输出到 Kafka
query = processedDF.selectExpr('CAST(id AS STRING) AS key', 'to_json(struct(*)) AS value') \
    .writeStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', 'localhost:9092') \
    .option('topic', 'output-topic') \
    .option('checkpointLocation', 'hdfs://path/to/checkpoint') \
    .start()

# 等待查询完成
query.awaitTermination()
```

## 10. 最佳实践

### 10.1 数据模式
- **始终为流数据指定明确的 schema**：避免使用 inferSchema，提高性能和可靠性
- **使用 StructType 定义 schema**：明确字段类型和结构
- **处理嵌套数据**：使用 from_json 等函数处理嵌套的 JSON 数据

### 10.2 性能优化
- **合理设置批处理间隔**：根据数据速率和处理复杂度调整
- **使用适当的输出模式**：根据业务需求选择 Append、Complete 或 Update 模式
- **优化查询计划**：使用 explain() 查看和优化查询计划
- **合理设置分区数**：根据集群资源和数据量调整
- **使用缓存**：对频繁使用的数据集进行缓存

### 10.3 容错
- **启用检查点**：确保故障恢复能力
- **选择可靠的数据源**：使用支持事务的数据源
- **处理重复数据**：实现幂等处理逻辑
- **监控检查点大小**：避免检查点过大影响性能

### 10.4 监控
- **监控流处理延迟**：使用 spark.sql.streaming.metricsEnabled 启用指标
- **监控处理速率**：跟踪每秒处理的记录数
- **监控状态大小**：避免状态过大导致内存问题
- **监控检查点操作**：确保检查点操作正常

### 10.5 部署
- **在生产环境中使用集群模式**：避免使用本地模式
- **合理配置资源**：根据数据量和处理需求配置 executor 内存和核心数
- **使用 YARN 或 Kubernetes**：在生产环境中使用集群管理器
- **配置日志级别**：适当设置日志级别，避免过多日志影响性能

### 10.6 常见问题
- **数据倾斜**：使用 salt 或重新分区解决
- **内存溢出**：调整 executor 内存和状态管理策略
- **检查点失败**：确保检查点目录有足够的空间和权限
- **处理延迟高**：优化查询逻辑和资源配置
- **状态大小增长**：实现状态清理和超时机制

## 11. 总结

Structured Streaming 是 Spark 生态系统中用于流处理的重要组件，提供了声明式的流处理 API，将流数据视为无限增长的表，支持与批处理相同的操作。本文档详细介绍了 Structured Streaming 的基本概念、流处理模型、数据源连接、流数据处理、输出操作、检查点、流处理控制、状态管理、示例应用以及最佳实践。

通过合理使用 Structured Streaming 的各种特性，可以构建高效、可靠的实时数据处理应用。它的容错机制和 exactly-once 语义保证了数据处理的一致性，而其与 Spark 其他组件的无缝集成使得构建端到端的大数据处理流水线变得更加简单。

掌握 Structured Streaming 的使用技巧，对于深入理解流处理和构建实时数据处理应用至关重要。通过本文档的学习，您应该能够在实际应用中灵活运用 Structured Streaming 处理和分析流数据，解决各种实时数据处理问题。