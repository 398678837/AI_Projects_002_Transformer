# MapReduce 计算详细文档

## 1. MapReduce 基本概念

MapReduce 是一种分布式计算框架，用于处理大规模数据集。它由 Google 提出，Hadoop 实现了开源版本。MapReduce 的核心思想是将计算分为 Map 和 Reduce 两个阶段，适合处理批处理任务。

### 1.1 设计理念
- **分而治之**：将大规模数据集分割成多个小数据集进行处理
- **移动计算比移动数据更高效**：将计算任务分发到数据所在的节点
- **容错性**：自动处理节点故障，确保作业正常运行
- **可扩展性**：可以线性扩展集群规模，处理 PB 级数据

### 1.2 适用场景
- **批处理任务**：如日志分析、数据挖掘、机器学习等
- **大规模数据处理**：处理 TB 或 PB 级数据
- **离线处理**：不适合实时处理任务

## 2. MapReduce 工作原理

MapReduce 作业执行过程分为三个主要阶段：Map 阶段、Shuffle 阶段和 Reduce 阶段。

### 2.1 Map 阶段
1. **输入数据分割**：将输入数据分割成多个 split（默认大小为 HDFS 块大小）
2. **Map 任务分配**：每个 split 由一个 Map 任务处理
3. **Map 函数执行**：将输入数据转换为 (key, value) 对
4. **本地排序**：Map 输出结果在本地进行排序

### 2.2 Shuffle 阶段
1. **数据传输**：将 Map 输出结果传输到 Reduce 节点
2. **排序和分组**：对 Map 输出进行排序和分组，将相同 key 的 value 汇总到一起
3. **数据分发**：将结果分发到对应的 Reduce 任务

### 2.3 Reduce 阶段
1. **Reduce 函数执行**：对每个 key 的 value 集合进行处理
2. **结果输出**：生成最终结果并输出到 HDFS

### 2.4 执行流程
1. **作业提交**：客户端提交 MapReduce 作业
2. **作业初始化**：ResourceManager 创建 JobTracker，分配作业 ID
3. **任务分配**：JobTracker 为作业分配 Map 和 Reduce 任务
4. **任务执行**：Worker 节点执行 Map 和 Reduce 任务
5. **结果汇总**：收集所有 Reduce 任务的输出结果

## 3. MapReduce 示例：单词计数

单词计数是 MapReduce 的经典示例，用于统计文本文件中每个单词出现的次数。

### 3.1 Map 函数
```python
def map(key, value):
    for word in value.split():
        emit(word, 1)
```

### 3.2 Reduce 函数
```python
def reduce(key, values):
    total = 0
    for v in values:
        total += v
    emit(key, total)
```

### 3.3 执行命令
```bash
hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount input output
```

### 3.4 执行流程
1. **输入**：文本文件
2. **Map 阶段**：将文本分割成单词，每个单词映射为 (word, 1)
3. **Shuffle 阶段**：将相同单词的 (word, 1) 对分组
4. **Reduce 阶段**：对每个单词的计数进行求和
5. **输出**：每个单词及其出现次数

## 4. MapReduce 配置

### 4.1 主要配置文件
- **mapred-site.xml**：MapReduce 特定配置
- **yarn-site.xml**：YARN 资源管理配置

### 4.2 关键配置参数
- **mapreduce.framework.name**：执行框架，默认为 yarn
- **mapreduce.map.memory.mb**：Map 任务内存，默认为 1024MB
- **mapreduce.reduce.memory.mb**：Reduce 任务内存，默认为 1024MB
- **mapreduce.map.cpu.vcores**：Map 任务 CPU 核心数，默认为 1
- **mapreduce.reduce.cpu.vcores**：Reduce 任务 CPU 核心数，默认为 1
- **mapreduce.job.reduces**：Reduce 任务数量，默认为 1
- **mapreduce.map.output.compress**：是否压缩 Map 输出，默认为 false
- **mapreduce.output.fileoutputformat.compress**：是否压缩最终输出，默认为 false

### 4.3 配置示例

**mapred-site.xml**：
```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
  <property>
    <name>mapreduce.map.memory.mb</name>
    <value>2048</value>
  </property>
  <property>
    <name>mapreduce.reduce.memory.mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>mapreduce.job.reduces</name>
    <value>4</value>
  </property>
</configuration>
```

## 5. MapReduce Java API

### 5.1 核心类
- **Mapper**：实现 Map 功能
- **Reducer**：实现 Reduce 功能
- **Job**：配置和提交 MapReduce 作业
- **Configuration**：配置信息
- **InputFormat**：输入格式处理
- **OutputFormat**：输出格式处理

### 5.2 示例代码

```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.Iterator;

public class WordCount {
    // Mapper 类
    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.split(" ");
            for (String w : words) {
                word.set(w);
                context.write(word, one);
            }
        }
    }

    // Reducer 类
    public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    // 主方法
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "WordCount");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

### 5.3 编译和运行
1. **编译**：`javac -cp $(hadoop classpath) WordCount.java`
2. **打包**：`jar cvf WordCount.jar WordCount*.class`
3. **运行**：`hadoop jar WordCount.jar WordCount input output`

## 6. MapReduce 作业提交

### 6.1 命令行提交
```bash
hadoop jar jarfile.MainClass input output
```

### 6.2 Java API 提交
```java
Job job = Job.getInstance(conf, "WordCount");
job.setJarByClass(WordCount.class);
job.setMapperClass(Map.class);
job.setReducerClass(Reduce.class);
job.setOutputKeyClass(Text.class);
job.setOutputValueClass(IntWritable.class);
FileInputFormat.addInputPath(job, new Path(args[0]));
FileOutputFormat.setOutputPath(job, new Path(args[1]));
System.exit(job.waitForCompletion(true) ? 0 : 1);
```

### 6.3 作业监控
- **YARN Web UI**：http://resourcemanager:8088
- **命令行**：`yarn application -list`
- **查看作业日志**：`yarn logs -applicationId <app_id>`

## 7. MapReduce 性能优化

### 7.1 输入数据优化
- **合理设置 split 大小**：根据文件大小和集群资源调整
- **使用 CombineFileInputFormat**：处理大量小文件，减少 Map 任务数量
- **数据预处理**：去除无用数据，减少处理量

### 7.2 Map 阶段优化
- **使用 Combiner**：在 Map 节点本地合并结果，减少网络传输
- **合理设置 Map 任务内存和 CPU**：根据任务复杂度调整
- **批量处理**：减少 I/O 操作次数

### 7.3 Shuffle 阶段优化
- **合理设置缓冲区大小**：`mapreduce.task.io.sort.mb`，默认为 100MB
- **使用压缩**：减少数据传输量，`mapreduce.map.output.compress=true`
- **合理设置排序溢出阈值**：`mapreduce.map.sort.spill.percent`，默认为 0.8

### 7.4 Reduce 阶段优化
- **合理设置 Reduce 任务数量**：通常为集群节点数的 1-2 倍
- **避免数据倾斜**：使用自定义 Partitioner 均匀分配数据
- **批量处理**：减少 I/O 操作次数
- **合理设置 Reduce 任务内存**：根据数据量调整

## 8. MapReduce 数据倾斜

### 8.1 数据倾斜的原因
- **某些 key 的数据量远大于其他 key**
- **导致某些 Reduce 任务处理时间过长**
- **整体作业执行时间取决于最慢的 Reduce 任务**

### 8.2 解决方案
- **数据预处理**：去除异常值，处理重复数据
- **使用随机前缀**：为热点 key 添加随机前缀，分散到不同的 Reduce 任务
- **自定义 Partitioner**：根据数据分布自定义分区策略，均匀分配数据
- **增加 Reduce 任务数量**：提高并行度，减轻单个 Reduce 任务的负担
- **使用 Combiner**：减少数据量，缓解数据倾斜
- **使用 MapJoin**：对于小表，使用 MapJoin 避免 Shuffle 过程

## 9. MapReduce 常见问题

### 9.1 作业失败
- **检查日志文件**：`yarn logs -applicationId <app_id>`
- **检查输入输出路径**：确保输入路径存在，输出路径不存在
- **检查权限**：确保用户有适当的权限
- **检查代码逻辑**：确保 Map 和 Reduce 函数逻辑正确

### 9.2 内存溢出
- **增加任务内存配置**：调整 `mapreduce.map.memory.mb` 和 `mapreduce.reduce.memory.mb`
- **优化代码**：减少内存使用，避免一次性加载大量数据
- **检查数据量**：确保数据量在预期范围内

### 9.3 执行速度慢
- **检查数据倾斜**：使用 `hadoop job -status <job_id>` 查看任务执行情况
- **优化作业配置**：调整任务数量、内存配置等
- **考虑使用更高效的计算框架**：如 Spark、Flink 等

### 9.4 其他常见问题
- **数据格式问题**：确保输入数据格式正确
- **网络问题**：确保集群网络畅通
- **磁盘空间不足**：监控和清理磁盘空间

## 10. MapReduce 最佳实践

### 10.1 代码设计
- **保持逻辑简单清晰**：Map 和 Reduce 函数逻辑应该简单明了
- **避免在 Map 或 Reduce 中执行 heavy 操作**：如数据库连接、复杂计算等
- **使用适当的数据类型**：根据数据特点选择合适的数据类型
- **处理异常情况**：编写健壮的代码，处理可能的异常

### 10.2 数据处理
- **预处理数据**：去除无用信息，减少处理量
- **使用适当的输入/输出格式**：根据数据特点选择合适的格式
- **数据压缩**：使用压缩减少存储和传输开销
- **合理设计 key**：确保 key 分布均匀，避免数据倾斜

### 10.3 作业配置
- **根据数据量和集群资源设置合理的任务数量**：避免任务过多或过少
- **监控作业执行情况**：及时调整配置
- **使用合适的调度策略**：根据作业优先级和资源需求选择调度策略
- **设置合理的超时时间**：避免作业无限期等待

### 10.4 集群管理
- **监控集群状态**：定期检查节点状态和资源使用情况
- **合理规划集群规模**：根据数据量和处理需求调整
- **定期维护**：清理日志，检查硬件状态
- **备份数据**：确保数据安全

## 11. 总结

MapReduce 是一种强大的分布式计算框架，为大规模数据处理提供了可靠的解决方案。通过合理的设计和优化，可以充分发挥 MapReduce 的性能，处理 PB 级数据。

随着大数据技术的发展，MapReduce 也在不断演进，同时也出现了更多的计算框架，如 Spark、Flink 等。这些框架在某些场景下比 MapReduce 更高效，但 MapReduce 作为大数据处理的基础框架，仍然具有重要的地位。

掌握 MapReduce 的原理和使用方法，对于理解分布式计算和大数据处理至关重要。通过不断学习和实践，可以更好地应用 MapReduce 解决实际问题，提高数据处理效率。