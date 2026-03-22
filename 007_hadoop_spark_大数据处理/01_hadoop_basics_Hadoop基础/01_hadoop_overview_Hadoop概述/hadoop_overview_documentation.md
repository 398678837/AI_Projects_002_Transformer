# Hadoop 概述详细文档

## 1. Hadoop 基本概念

Hadoop 是一个开源的分布式计算框架，专为处理大规模数据集而设计。它由 Apache 基金会开发和维护，是大数据生态系统的核心组件。

### 1.1 设计理念
- **移动计算比移动数据更高效**：将计算任务分发到数据所在的节点，减少网络传输开销
- **高可靠性**：通过多副本存储确保数据安全
- **高扩展性**：可以线性扩展集群规模，支持处理 PB 级数据
- **容错性**：自动处理节点故障，确保作业正常运行

### 1.2 主要组件
- **HDFS**：分布式文件系统，负责数据存储
- **MapReduce**：分布式计算框架，负责数据处理
- **YARN**：资源管理系统，负责资源分配和作业调度

## 2. Hadoop 架构

Hadoop 采用主从架构（Master-Slave Architecture），主要由以下组件组成：

### 2.1 HDFS 架构
- **NameNode**：主节点，负责管理文件系统命名空间和元数据
- **DataNode**：从节点，负责存储实际数据块
- **Secondary NameNode**：辅助节点，负责定期合并编辑日志，减轻 NameNode 负担

### 2.2 YARN 架构
- **ResourceManager**：主节点，负责集群资源管理和作业调度
- **NodeManager**：从节点，负责单个节点的资源管理和容器监控
- **ApplicationMaster**：每个应用的主进程，负责协调应用的执行

### 2.3 MapReduce 架构
- **JobTracker**：负责作业的调度和监控（Hadoop 1.x）
- **TaskTracker**：负责执行具体的 Map 和 Reduce 任务（Hadoop 1.x）
- **在 Hadoop 2.x 中**：MapReduce 作业由 YARN 管理

## 3. Hadoop 生态系统

Hadoop 生态系统包含多个组件，共同构成了完整的大数据处理解决方案：

### 3.1 存储组件
- **HDFS**：分布式文件系统，适合存储大规模数据
- **HBase**：分布式列存储数据库，适合实时随机读写

### 3.2 计算组件
- **MapReduce**：批处理框架，适合大规模数据处理
- **Spark**：快速计算引擎，支持批处理、流处理和机器学习
- **Tez**：基于 YARN 的计算框架，提供更灵活的数据流处理

### 3.3 数据管理组件
- **Hive**：数据仓库工具，提供类 SQL 查询接口
- **Pig**：数据流处理工具，使用 Pig Latin 语言
- **Mahout**：机器学习库，提供常见的机器学习算法

### 3.4 数据集成组件
- **Flume**：日志收集工具，用于将数据从各种来源传输到 HDFS
- **Sqoop**：数据导入导出工具，用于在 Hadoop 和关系型数据库之间传输数据
- **Kafka**：分布式消息队列，用于高吞吐量的数据流处理

## 4. Hadoop 安装和配置

### 4.1 前提条件
- Java JDK 1.8 或更高版本
- 足够的磁盘空间和内存
- 网络连接正常

### 4.2 安装步骤
1. **下载 Hadoop**：从 https://hadoop.apache.org/releases.html 下载最新版本
2. **解压安装包**：`tar -xzvf hadoop-3.x.x.tar.gz`
3. **配置环境变量**：
   ```bash
   export HADOOP_HOME=/path/to/hadoop
   export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
   ```
4. **配置文件**：
   - `core-site.xml`：核心配置
   - `hdfs-site.xml`：HDFS 配置
   - `yarn-site.xml`：YARN 配置
   - `mapred-site.xml`：MapReduce 配置

### 4.3 配置示例

**core-site.xml**：
```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

**hdfs-site.xml**：
```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>/path/to/hadoop/data/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/path/to/hadoop/data/datanode</value>
  </property>
</configuration>
```

## 5. Hadoop 集群管理

### 5.1 启动和停止服务
- **启动 HDFS**：`start-dfs.sh`
- **启动 YARN**：`start-yarn.sh`
- **启动所有服务**：`start-all.sh`
- **停止 HDFS**：`stop-dfs.sh`
- **停止 YARN**：`stop-yarn.sh`
- **停止所有服务**：`stop-all.sh`

### 5.2 集群状态检查
- **查看 HDFS 状态**：`hdfs dfsadmin -report`
- **查看 YARN 状态**：`yarn node -list`
- **查看作业状态**：`yarn application -list`

## 6. Hadoop 命令行工具

### 6.1 HDFS 命令
- **创建目录**：`hdfs dfs -mkdir /path`
- **上传文件**：`hdfs dfs -put localfile /hdfs/path`
- **下载文件**：`hdfs dfs -get /hdfs/path localfile`
- **查看文件**：`hdfs dfs -cat /hdfs/path`
- **列出目录**：`hdfs dfs -ls /path`
- **删除文件**：`hdfs dfs -rm /hdfs/path`
- **删除目录**：`hdfs dfs -rm -r /hdfs/path`

### 6.2 MapReduce 命令
- **运行作业**：`hadoop jar jarfile.MainClass input output`
- **查看作业历史**：`mapred job -history /history/path`

### 6.3 YARN 命令
- **查看应用**：`yarn application -list`
- **杀死应用**：`yarn application -kill application_id`
- **查看节点**：`yarn node -list`

## 7. Hadoop 示例应用

### 7.1 单词计数 (WordCount)
- **功能**：统计文本文件中每个单词出现的次数
- **输入**：文本文件
- **输出**：每个单词及其出现次数
- **命令**：
  ```bash
  hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount input output
  ```

### 7.2 排序 (Sort)
- **功能**：对输入的键值对进行排序
- **输入**：键值对
- **输出**：排序后的键值对
- **命令**：
  ```bash
  hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar sort input output
  ```

### 7.3 其他示例
- **Pi 估算**：计算 π 的近似值
- **二次排序**：对键值对进行二次排序
- **矩阵乘法**：执行矩阵乘法操作

## 8. Hadoop Web UI

Hadoop 提供了 Web UI 用于监控和管理集群：

### 8.1 HDFS Web UI
- **地址**：http://namenode:9870
- **功能**：查看文件系统状态、浏览文件、查看 DataNode 状态

### 8.2 YARN Web UI
- **地址**：http://resourcemanager:8088
- **功能**：查看集群资源使用情况、监控作业执行状态

### 8.3 MapReduce JobHistory
- **地址**：http://jobhistoryserver:19888
- **功能**：查看已完成作业的历史记录

## 9. Hadoop 最佳实践

### 9.1 数据管理
- **合理设置块大小**：默认为 128MB，大文件可以设置更大
- **合理设置副本数**：默认为 3，根据数据重要性和存储成本调整
- **使用数据压缩**：减少网络传输和存储开销，推荐使用 Snappy 或 LZ4 压缩
- **合理规划目录结构**：便于数据管理和访问

### 9.2 作业优化
- **数据本地化**：确保计算任务调度到数据所在节点
- **合理设置 Map 和 Reduce 任务数**：根据数据量和集群资源调整
- **使用 Combiner**：减少 Map 到 Reduce 的数据传输
- **使用适当的 InputFormat 和 OutputFormat**：根据数据类型选择合适的格式

### 9.3 集群管理
- **监控集群状态**：定期检查节点状态和资源使用情况
- **合理规划集群规模**：根据数据量和处理需求调整
- **定期备份**：确保数据安全
- **使用自动化工具**：如 Ansible 或 Puppet 管理集群配置

## 10. Hadoop 常见问题

### 10.1 权限问题
- **症状**：作业执行失败，出现权限错误
- **解决方案**：确保用户有适当的权限，检查 HDFS 权限设置

### 10.2 内存不足
- **症状**：作业执行失败，出现 OutOfMemoryError
- **解决方案**：调整 JVM 堆大小，检查数据量是否过大

### 10.3 网络问题
- **症状**：节点间通信失败，作业执行缓慢
- **解决方案**：确保集群网络畅通，检查防火墙设置

### 10.4 磁盘空间不足
- **症状**：HDFS 写操作失败，出现磁盘空间不足错误
- **解决方案**：监控和清理磁盘空间，增加存储容量

### 10.5 配置错误
- **症状**：服务启动失败或作业执行异常
- **解决方案**：检查配置文件是否正确，确保配置项格式正确

## 11. 总结

Hadoop 是一个强大的分布式计算框架，为大数据处理提供了可靠的解决方案。通过合理的安装配置和优化，可以充分发挥 Hadoop 的性能，处理大规模数据集。

随着大数据技术的发展，Hadoop 生态系统不断丰富，为不同场景提供了更多的工具和解决方案。掌握 Hadoop 及其生态系统，对于处理和分析大规模数据至关重要。