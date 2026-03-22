# YARN 资源管理详细文档

## 1. YARN 基本概念

YARN（Yet Another Resource Negotiator）是 Hadoop 的资源管理系统，负责管理集群中的资源（CPU、内存）和调度作业。它支持多种计算框架，如 MapReduce、Spark、Tez 等。

### 1.1 设计目标
- **提高集群利用率**：更有效地利用集群资源
- **支持多种计算框架**：不限于 MapReduce，还支持 Spark、Tez 等
- **提高集群可扩展性**：支持更大规模的集群
- **提高集群可靠性**：提供高可用机制

### 1.2 核心组件
- **ResourceManager (RM)**：集群资源管理和作业调度
- **NodeManager (NM)**：节点资源管理
- **ApplicationMaster (AM)**：应用管理
- **Container**：资源分配的基本单位

## 2. YARN 架构

YARN 采用主从架构，主要由以下组件组成：

### 2.1 ResourceManager (RM)
- **功能**：集群资源管理和作业调度
- **职责**：
  - 处理客户端请求
  - 分配资源给应用
  - 监控 NodeManager
  - 调度作业执行
  - 管理集群资源

### 2.2 NodeManager (NM)
- **功能**：节点资源管理
- **职责**：
  - 管理节点上的资源（CPU、内存）
  - 启动和监控容器
  - 向 ResourceManager 报告节点状态
  - 管理节点上的应用程序

### 2.3 ApplicationMaster (AM)
- **功能**：应用管理
- **职责**：
  - 为应用申请资源
  - 协调任务执行
  - 监控任务状态
  - 处理任务失败

### 2.4 Container
- **功能**：资源分配的基本单位
- **特性**：
  - 包含 CPU 和内存资源
  - 运行任务的环境
  - 由 NodeManager 管理
  - 是资源隔离的基本单位

## 3. YARN 工作流程

YARN 作业执行的典型流程如下：

1. **客户端提交应用**：客户端向 ResourceManager 提交应用
2. **分配第一个容器**：ResourceManager 为 ApplicationMaster 分配第一个容器
3. **启动 ApplicationMaster**：NodeManager 在分配的容器中启动 ApplicationMaster
4. **ApplicationMaster 注册**：ApplicationMaster 向 ResourceManager 注册
5. **申请资源**：ApplicationMaster 根据应用需求向 ResourceManager 申请资源
6. **分配资源**：ResourceManager 分配资源并通知 NodeManager
7. **创建容器**：NodeManager 创建容器并启动任务
8. **监控任务**：ApplicationMaster 监控任务执行情况
9. **任务完成**：任务执行完成后，ApplicationMaster 向 ResourceManager 注销
10. **释放资源**：ResourceManager 释放资源

## 4. YARN 配置

### 4.1 主要配置文件
- **yarn-site.xml**：YARN 核心配置文件
- **capacity-scheduler.xml**：Capacity 调度器配置
- **fair-scheduler.xml**：Fair 调度器配置

### 4.2 关键配置参数
- **yarn.resourcemanager.hostname**：ResourceManager 主机名
- **yarn.nodemanager.resource.memory-mb**：节点内存总量
- **yarn.nodemanager.resource.cpu-vcores**：节点 CPU 核心数
- **yarn.scheduler.minimum-allocation-mb**：最小内存分配
- **yarn.scheduler.maximum-allocation-mb**：最大内存分配
- **yarn.scheduler.minimum-allocation-vcores**：最小 CPU 分配
- **yarn.scheduler.maximum-allocation-vcores**：最大 CPU 分配
- **yarn.resourcemanager.scheduler.class**：调度器类

### 4.3 配置示例

**yarn-site.xml**：
```xml
<configuration>
  <property>
    <name>yarn.resourcemanager.hostname</name>
    <value>resourcemanager</value>
  </property>
  <property>
    <name>yarn.nodemanager.resource.memory-mb</name>
    <value>8192</value>
  </property>
  <property>
    <name>yarn.nodemanager.resource.cpu-vcores</name>
    <value>4</value>
  </property>
  <property>
    <name>yarn.scheduler.minimum-allocation-mb</name>
    <value>1024</value>
  </property>
  <property>
    <name>yarn.scheduler.maximum-allocation-mb</name>
    <value>4096</value>
  </property>
  <property>
    <name>yarn.scheduler.minimum-allocation-vcores</name>
    <value>1</value>
  </property>
  <property>
    <name>yarn.scheduler.maximum-allocation-vcores</name>
    <value>2</value>
  </property>
</configuration>
```

## 5. YARN 命令行工具

### 5.1 应用管理
- **列出应用**：`yarn application -list`
- **查看应用状态**：`yarn application -status <app_id>`
- **杀死应用**：`yarn application -kill <app_id>`
- **查看应用历史**：`yarn application -list -appStates FINISHED`

### 5.2 节点管理
- **列出节点**：`yarn node -list`
- **查看节点状态**：`yarn node -status <node_id>`
- **查看节点健康状态**：`yarn node -list -states HEALTHY`

### 5.3 队列管理
- **查看队列**：`yarn queue -status <queue_name>`
- **查看所有队列**：`yarn queue -list`

### 5.4 日志查看
- **查看应用日志**：`yarn logs -applicationId <app_id>`
- **查看容器日志**：`yarn logs -containerId <container_id>`
- **查看特定节点的日志**：`yarn logs -applicationId <app_id> -nodeId <node_id>`

### 5.5 其他命令
- **查看集群状态**：`yarn cluster -status`
- **实时查看应用资源使用情况**：`yarn top`

## 6. YARN 调度器

YARN 支持三种调度器：FIFO 调度器、Capacity 调度器和 Fair 调度器。

### 6.1 FIFO 调度器
- **特点**：先进先出，简单但可能导致资源利用率低
- **适用场景**：单用户环境，简单的工作负载
- **配置**：默认调度器

### 6.2 Capacity 调度器
- **特点**：
  - 为不同队列分配固定容量
  - 支持队列层级和资源共享
  - 支持资源预留
  - 适合多用户环境
- **适用场景**：多用户环境，需要资源隔离的场景
- **配置文件**：`capacity-scheduler.xml`

### 6.3 Fair 调度器
- **特点**：
  - 公平分配资源
  - 支持资源抢占
  - 支持队列层级
  - 适合多用户环境
- **适用场景**：多用户环境，需要公平分配资源的场景
- **配置文件**：`fair-scheduler.xml`

## 7. YARN 队列配置

### 7.1 Capacity 调度器队列配置

**capacity-scheduler.xml** 配置示例：
```xml
<configuration>
  <property>
    <name>yarn.scheduler.capacity.root.queues</name>
    <value>default,production,development</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.default.capacity</name>
    <value>40</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.production.capacity</name>
    <value>40</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.development.capacity</name>
    <value>20</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.production.maximum-capacity</name>
    <value>80</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.default.maximum-capacity</name>
    <value>80</value>
  </property>
  <property>
    <name>yarn.scheduler.capacity.root.development.maximum-capacity</name>
    <value>40</value>
  </property>
</configuration>
```

### 7.2 Fair 调度器队列配置

**fair-scheduler.xml** 配置示例：
```xml
<allocations>
  <queue name="root">
    <queue name="default">
      <weight>40</weight>
      <maxResources>80%</maxResources>
    </queue>
    <queue name="production">
      <weight>40</weight>
      <maxResources>80%</maxResources>
    </queue>
    <queue name="development">
      <weight>20</weight>
      <maxResources>40%</maxResources>
    </queue>
  </queue>
</allocations>
```

## 8. YARN 高可用

### 8.1 高可用架构
YARN 高可用通过配置多个 ResourceManager（Active/Standby）实现，使用 Zookeeper 进行状态管理和故障转移。

### 8.2 关键配置

**yarn-site.xml** 高可用配置：
```xml
<configuration>
  <property>
    <name>yarn.resourcemanager.ha.enabled</name>
    <value>true</value>
  </property>
  <property>
    <name>yarn.resourcemanager.cluster-id</name>
    <value>cluster1</value>
  </property>
  <property>
    <name>yarn.resourcemanager.ha.rm-ids</name>
    <value>rm1,rm2</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname.rm1</name>
    <value>resourcemanager1</value>
  </property>
  <property>
    <name>yarn.resourcemanager.hostname.rm2</name>
    <value>resourcemanager2</value>
  </property>
  <property>
    <name>yarn.resourcemanager.zk-address</name>
    <value>zookeeper1:2181,zookeeper2:2181,zookeeper3:2181</value>
  </property>
</configuration>
```

### 8.3 故障转移
当 Active ResourceManager 故障时，Standby ResourceManager 会通过 Zookeeper 检测到故障并自动切换为 Active 状态，确保集群正常运行。

## 9. YARN 监控

### 9.1 Web UI
- **ResourceManager Web UI**：http://rm_host:8088
  - 查看集群资源使用情况
  - 查看运行中的应用
  - 查看应用历史
  - 查看节点状态

- **NodeManager Web UI**：http://nm_host:8042
  - 查看节点资源使用情况
  - 查看节点上的容器
  - 查看节点日志

### 9.2 命令行监控
- **实时查看应用资源使用情况**：`yarn top`
- **查看运行中的应用**：`yarn application -list -appStates RUNNING`
- **查看节点状态**：`yarn node -list`
- **查看集群状态**：`yarn cluster -status`

### 9.3 日志监控
- **YARN 日志目录**：$HADOOP_HOME/logs/
- **应用日志**：通过 Web UI 或 `yarn logs` 命令查看
- **节点日志**：NodeManager 日志目录

### 9.4 第三方监控工具
- **Ambari**：Hadoop 集群管理和监控工具
- **Cloudera Manager**：Cloudera 发行版的集群管理工具
- **Ganglia**：分布式监控系统
- **Nagios**：网络监控工具

## 10. YARN 最佳实践

### 10.1 资源配置
- **根据节点硬件配置设置合理的资源总量**：
  - 内存：`yarn.nodemanager.resource.memory-mb`
  - CPU：`yarn.nodemanager.resource.cpu-vcores`
- **为应用设置适当的资源请求**：
  - 避免资源请求过大或过小
  - 根据应用实际需求调整
- **避免过度分配资源**：
  - 预留部分资源给系统使用
  - 考虑其他服务的资源需求

### 10.2 调度器选择
- **单用户环境**：使用 FIFO 调度器
- **多用户环境**：使用 Capacity 或 Fair 调度器
- **关键任务**：使用 Capacity 调度器并设置专用队列
- **公平分配**：使用 Fair 调度器

### 10.3 队列配置
- **根据业务需求设置队列**：
  - 生产队列：分配更多资源
  - 开发队列：分配较少资源
  - 测试队列：分配适量资源
- **设置合理的队列容量**：
  - 避免队列容量过小导致任务等待
  - 避免队列容量过大导致资源浪费

### 10.4 监控和调优
- **定期监控集群资源使用情况**：
  - 使用 Web UI 或命令行工具
  - 关注资源利用率和任务执行情况
- **根据应用特点调整配置**：
  - 内存密集型应用：增加内存分配
  - CPU 密集型应用：增加 CPU 分配
- **及时清理失败的应用**：
  - 避免失败应用占用资源
  - 使用 `yarn application -kill` 命令清理

### 10.5 高可用配置
- **配置 ResourceManager 高可用**：
  - 避免单点故障
  - 确保集群稳定运行
- **使用 Zookeeper 进行状态管理**：
  - 确保故障转移的可靠性
  - 监控 Zookeeper 状态

## 11. 常见问题和解决方案

### 11.1 资源分配问题
- **症状**：应用无法获取足够的资源
- **解决方案**：
  - 检查集群资源使用情况
  - 调整应用资源请求
  - 增加集群资源

### 11.2 应用执行缓慢
- **症状**：应用执行时间过长
- **解决方案**：
  - 检查数据倾斜
  - 优化应用代码
  - 调整资源分配
  - 检查集群负载

### 11.3 节点故障
- **症状**：节点不可用
- **解决方案**：
  - 检查节点状态
  - 检查网络连接
  - 重启 NodeManager
  - 检查硬件故障

### 11.4 资源管理器故障
- **症状**：ResourceManager 不可用
- **解决方案**：
  - 检查 ResourceManager 状态
  - 重启 ResourceManager
  - 启用高可用

### 11.5 调度器问题
- **症状**：任务调度不合理
- **解决方案**：
  - 调整调度器配置
  - 检查队列容量
  - 选择合适的调度器

## 12. 总结

YARN 是 Hadoop 生态系统中的核心组件，为大数据处理提供了强大的资源管理和调度能力。通过合理的配置和管理，可以充分发挥 YARN 的性能，提高集群利用率，支持多种计算框架。

随着大数据技术的发展，YARN 也在不断演进，引入了更多的特性和改进，如弹性资源管理、容器重用等，以满足不断增长的计算需求。掌握 YARN 的使用和管理，对于构建和维护大数据系统至关重要。

通过本文档的学习，您应该对 YARN 的基本概念、架构、工作流程、配置和最佳实践有了全面的了解。在实际应用中，您可以根据具体的业务需求和集群环境，对 YARN 进行合理的配置和调优，以达到最佳的性能和可靠性。