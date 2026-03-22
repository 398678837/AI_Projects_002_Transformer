# HDFS 分布式存储详细文档

## 1. HDFS 基本概念

HDFS（Hadoop Distributed File System）是 Hadoop 的分布式文件系统，专为存储大规模数据集而设计。它提供了高可靠性、高吞吐量和可扩展性的存储解决方案。

### 1.1 核心概念
- **块（Block）**：HDFS 中数据存储的基本单位，默认为 128MB
- **副本（Replica）**：数据块的多个副本，默认为 3 个，存储在不同的 DataNode 上
- **命名空间（Namespace）**：HDFS 的文件系统目录结构
- **元数据（Metadata）**：关于文件和目录的信息，如权限、大小、位置等

### 1.2 设计目标
- **处理大规模数据集**：支持存储和处理 PB 级数据
- **高可靠性**：通过多副本存储确保数据安全
- **高吞吐量**：优化大规模数据的读写操作
- **可扩展性**：支持横向扩展，增加节点即可增加存储容量

## 2. HDFS 架构

HDFS 采用主从架构，主要由以下组件组成：

### 2.1 主节点（NameNode）
- **功能**：管理文件系统命名空间，维护文件和目录的元数据
- **职责**：
  - 管理文件系统的目录结构
  - 记录文件到数据块的映射关系
  - 管理 DataNode 的状态
  - 处理客户端的文件操作请求

### 2.2 从节点（DataNode）
- **功能**：存储实际的数据块，处理客户端的读写请求
- **职责**：
  - 存储数据块
  - 执行数据块的读写操作
  - 定期向 NameNode 发送心跳和块报告

### 2.3 辅助节点（SecondaryNameNode）
- **功能**：辅助 NameNode 进行 checkpoint，提高可靠性
- **职责**：
  - 定期合并编辑日志（EditLog）和命名空间镜像（FSImage）
  - 减轻 NameNode 的负担
  - 提供 NameNode 故障恢复的 checkpoint

### 2.4 数据块存储
- **块大小**：默认为 128MB，可根据文件大小和存储需求调整
- **副本策略**：
  - 第一个副本：存储在客户端所在节点
  - 第二个副本：存储在不同机架的节点
  - 第三个副本：存储在同一机架的不同节点

## 3. HDFS 配置

### 3.1 主要配置文件
- **core-site.xml**：核心配置，包含 HDFS 的基本设置
- **hdfs-site.xml**：HDFS 特定配置，包含存储和副本设置

### 3.2 关键配置参数
- **dfs.replication**：副本数，默认为 3
- **dfs.blocksize**：块大小，默认为 134217728（128MB）
- **dfs.namenode.name.dir**：NameNode 元数据存储目录
- **dfs.datanode.data.dir**：DataNode 数据存储目录
- **dfs.namenode.http-address**：NameNode Web UI 地址
- **dfs.datanode.http-address**：DataNode Web UI 地址

### 3.3 配置示例

**core-site.xml**：
```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>/path/to/hadoop/tmp</value>
  </property>
</configuration>
```

**hdfs-site.xml**：
```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>3</value>
  </property>
  <property>
    <name>dfs.blocksize</name>
    <value>134217728</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>/path/to/hadoop/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>/path/to/hadoop/datanode</value>
  </property>
</configuration>
```

## 4. HDFS 命令行操作

### 4.1 目录操作
- **创建目录**：`hdfs dfs -mkdir /path`
- **创建多级目录**：`hdfs dfs -mkdir -p /path/to/dir`
- **列出目录**：`hdfs dfs -ls /path`
- **递归列出目录**：`hdfs dfs -ls -R /path`
- **查看目录大小**：`hdfs dfs -du -h /path`

### 4.2 文件操作
- **上传文件**：`hdfs dfs -put localfile /hdfs/path`
- **下载文件**：`hdfs dfs -get /hdfs/path localfile`
- **查看文件**：`hdfs dfs -cat /hdfs/path`
- **复制文件**：`hdfs dfs -cp /hdfs/source /hdfs/destination`
- **移动文件**：`hdfs dfs -mv /hdfs/source /hdfs/destination`
- **删除文件**：`hdfs dfs -rm /hdfs/path`
- **删除目录**：`hdfs dfs -rm -r /hdfs/path`
- **查看文件大小**：`hdfs dfs -du -h /hdfs/path`

### 4.3 权限操作
- **更改权限**：`hdfs dfs -chmod 755 /hdfs/path`
- **更改所有者**：`hdfs dfs -chown user:group /hdfs/path`
- **更改所属组**：`hdfs dfs -chgrp group /hdfs/path`

### 4.4 系统操作
- **检查 HDFS 状态**：`hdfs dfsadmin -report`
- **进入安全模式**：`hdfs dfsadmin -safemode enter`
- **离开安全模式**：`hdfs dfsadmin -safemode leave`
- **检查文件系统**：`hdfs fsck /path`
- **启动均衡器**：`hdfs dfsadmin -balancer`

## 5. HDFS API

### 5.1 Java API

**核心类**：
- `FileSystem`：文件系统操作的抽象类
- `Path`：表示 HDFS 中的路径
- `FSDataInputStream`：读取文件
- `FSDataOutputStream`：写入文件
- `Configuration`：配置类

**示例代码**：
```java
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;

public class HDFSExample {
    public static void main(String[] args) throws Exception {
        // 创建配置对象
        Configuration conf = new Configuration();
        // 获取文件系统实例
        FileSystem fs = FileSystem.get(conf);
        
        // 检查文件是否存在
        Path path = new Path("/hdfs/path");
        if (fs.exists(path)) {
            System.out.println("File exists");
        } else {
            System.out.println("File does not exist");
        }
        
        // 读取文件
        if (fs.exists(path)) {
            FSDataInputStream in = fs.open(path);
            byte[] buffer = new byte[1024];
            int bytesRead = in.read(buffer);
            System.out.println(new String(buffer, 0, bytesRead));
            in.close();
        }
        
        // 写入文件
        FSDataOutputStream out = fs.create(new Path("/hdfs/newfile"));
        out.write("Hello HDFS".getBytes());
        out.close();
        
        // 关闭文件系统
        fs.close();
    }
}
```

### 5.2 Python API

**使用 hdfs3 库**：
```python
from hdfs3 import HDFileSystem

# 创建 HDFS 客户端
hdfs = HDFileSystem(host='namenode', port=9000)

# 列出目录
print(hdfs.ls('/'))

# 上传文件
hdfs.put('localfile', '/hdfs/path')

# 下载文件
hdfs.get('/hdfs/path', 'localfile')

# 读取文件
with hdfs.open('/hdfs/path', 'rb') as f:
    content = f.read()
    print(content)

# 写入文件
with hdfs.open('/hdfs/newfile', 'wb') as f:
    f.write(b'Hello HDFS')

# 检查文件是否存在
print(hdfs.exists('/hdfs/path'))

# 删除文件
hdfs.rm('/hdfs/path')
```

**使用 hdfs 库**：
```python
from hdfs import InsecureClient

# 创建 HDFS 客户端
client = InsecureClient('http://namenode:9870', user='hadoop')

# 列出目录
print(client.list('/'))

# 上传文件
client.upload('/hdfs/path', 'localfile')

# 下载文件
client.download('/hdfs/path', 'localfile')

# 读取文件
with client.read('/hdfs/path') as f:
    content = f.read()
    print(content)

# 写入文件
client.write('/hdfs/newfile', b'Hello HDFS')
```

## 6. HDFS 数据均衡

### 6.1 数据均衡的必要性
- 当集群中添加新节点时，需要将数据均匀分布到新节点
- 当节点存储使用不均匀时，需要重新平衡数据分布
- 数据均衡可以提高集群的整体性能和可靠性

### 6.2 启动均衡器
```bash
# 启动均衡器
hdfs dfsadmin -balancer

# 设置均衡阈值（默认 10%）
hdfs dfsadmin -balancer -threshold 5

# 取消均衡
hdfs dfsadmin -balancer -cancel
```

### 6.3 均衡器参数
- **threshold**：均衡阈值，默认为 10%，表示节点间存储使用率的最大差异
- **bandwidth**：均衡带宽，默认为 1MB/s，可以通过 `dfs.datanode.balance.bandwidthPerSec` 配置

## 7. HDFS 安全模式

### 7.1 安全模式概述
安全模式是 HDFS 的一种特殊状态，在这种状态下：
- 只允许读操作，不允许写操作
- NameNode 正在加载元数据并检查数据块的完整性
- 当集群启动时，会自动进入安全模式

### 7.2 安全模式操作
- **检查安全模式状态**：`hdfs dfsadmin -safemode get`
- **进入安全模式**：`hdfs dfsadmin -safemode enter`
- **离开安全模式**：`hdfs dfsadmin -safemode leave`
- **等待安全模式**：`hdfs dfsadmin -safemode wait`

## 8. HDFS 故障处理

### 8.1 NameNode 故障
- **原因**：硬件故障、软件错误、网络问题等
- **解决方案**：
  - 使用 SecondaryNameNode 的 checkpoint 恢复
  - 配置 NameNode 高可用（HA）
  - 定期备份 NameNode 元数据

### 8.2 DataNode 故障
- **原因**：硬件故障、网络问题、磁盘损坏等
- **解决方案**：
  - HDFS 会自动检测 DataNode 故障
  - 自动将故障节点上的数据块复制到其他节点
  - 检查节点状态：`hdfs dfsadmin -report`

### 8.3 数据块损坏
- **原因**：磁盘损坏、网络传输错误等
- **解决方案**：
  - HDFS 会自动检测损坏的数据块
  - 从其他副本恢复损坏的数据块
  - 手动检查：`hdfs fsck /path`

### 8.4 常见错误及解决方法
- **NameNode 无法启动**：检查配置文件和元数据目录
- **DataNode 无法连接到 NameNode**：检查网络连接和防火墙设置
- **文件读写失败**：检查权限和磁盘空间
- **均衡器无法启动**：检查集群状态和网络连接

## 9. HDFS 最佳实践

### 9.1 存储优化
- **合理设置块大小**：
  - 大文件：128MB 或 256MB
  - 小文件：考虑使用 SequenceFile 或 Avro
- **合理设置副本数**：
  - 关键数据：3 个或更多副本
  - 非关键数据：1-2 个副本
- **避免小文件**：
  - 小文件会增加 NameNode 内存开销
  - 使用 HAR (Hadoop Archive) 或 SequenceFile 合并小文件
- **数据压缩**：
  - 减少存储空间和网络传输
  - 常用压缩格式：Gzip, Snappy, LZO

### 9.2 性能优化
- **数据本地化**：将计算任务调度到数据所在节点
- **批量操作**：减少客户端与 NameNode 的交互次数
- **合理使用缓存**：对于频繁访问的数据，使用 HDFS 缓存
- **调整 JVM 参数**：根据服务器配置调整 NameNode 和 DataNode 的 JVM 参数

### 9.3 管理最佳实践
- **定期备份**：备份 NameNode 元数据
- **监控集群**：使用 Ambari 或 Cloudera Manager 监控集群状态
- **定期检查**：使用 `hdfs fsck` 定期检查文件系统
- **规划存储**：根据数据增长趋势规划存储容量
- **升级策略**：制定合理的集群升级策略

## 10. 总结

HDFS 是 Hadoop 生态系统的核心组件，为大数据处理提供了可靠的分布式存储解决方案。通过合理的配置和管理，可以充分发挥 HDFS 的性能和可靠性，为大规模数据处理提供有力支持。

随着大数据技术的发展，HDFS 也在不断演进，引入了更多的特性和改进，如 Erasure Coding、Federation、High Availability 等，以满足不断增长的存储需求。掌握 HDFS 的使用和管理，对于构建和维护大数据系统至关重要。