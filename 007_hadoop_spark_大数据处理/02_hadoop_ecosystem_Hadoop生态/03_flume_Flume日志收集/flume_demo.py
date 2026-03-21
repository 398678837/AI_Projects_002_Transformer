#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flume 日志收集演示

本脚本演示 Flume 的基本概念、配置和使用方法。
"""

import os
import sys

print("Flume 日志收集演示")
print("=" * 50)

# 1. Flume 基本概念
def flume_basics():
    print("\n1. Flume 基本概念:")
    print("- Flume 是 Apache 旗下的分布式日志收集系统")
    print("- 用于从不同数据源收集、聚合和传输大量日志数据")
    print("- 设计用于高可靠性、高容错性和可扩展性")
    print("- 支持多种数据源和目的地")

# 2. Flume 架构
def flume_architecture():
    print("\n2. Flume 架构:")
    print("- Agent: Flume 的基本运行单位")
    print("  - Source: 数据源，负责接收数据")
    print("  - Channel: 数据通道，存储数据")
    print("  - Sink: 数据目的地，将数据发送到目标系统")
    print("- Event: Flume 中的基本数据单元")
    print("- Interceptor: 拦截器，用于修改或过滤事件")
    print("- Channel Selector: 通道选择器，决定事件发送到哪个通道")
    print("- Sink Processor: 下沉处理器，管理多个 Sink")

# 3. Flume 配置文件
def flume_configuration():
    print("\n3. Flume 配置文件:")
    print("- 配置文件格式: 基于属性的配置")
    print("- 示例配置 (单 Agent):")
    print("  # 命名 Agent")
    print("  agent.sources = r1")
    print("  agent.sinks = k1")
    print("  agent.channels = c1")
    print("  ")
    print("  # 配置 Source")
    print("  agent.sources.r1.type = netcat")
    print("  agent.sources.r1.bind = localhost")
    print("  agent.sources.r1.port = 44444")
    print("  ")
    print("  # 配置 Sink")
    print("  agent.sinks.k1.type = logger")
    print("  ")
    print("  # 配置 Channel")
    print("  agent.channels.c1.type = memory")
    print("  agent.channels.c1.capacity = 1000")
    print("  agent.channels.c1.transactionCapacity = 100")
    print("  ")
    print("  # 绑定 Source、Channel 和 Sink")
    print("  agent.sources.r1.channels = c1")
    print("  agent.sinks.k1.channel = c1")

# 4. Flume 启动和停止
def flume_start_stop():
    print("\n4. Flume 启动和停止:")
    print("- 启动 Flume Agent:")
    print("  bin/flume-ng agent --conf conf --conf-file conf/flume.conf --name agent -Dflume.root.logger=INFO,console")
    print("- 后台启动 Flume Agent:")
    print("  nohup bin/flume-ng agent --conf conf --conf-file conf/flume.conf --name agent -Dflume.root.logger=INFO,console > flume.log 2>&1 &")
    print("- 停止 Flume Agent:")
    print("  ps aux | grep flume")
    print("  kill <pid>")

# 5. Flume 常用 Source
def flume_sources():
    print("\n5. Flume 常用 Source:")
    print("- Netcat Source: 监听网络端口，接收 TCP 数据")
    print("  type = netcat")
    print("  bind = localhost")
    print("  port = 44444")
    print("- Exec Source: 执行命令，收集命令输出")
    print("  type = exec")
    print("  command = tail -F /var/log/syslog")
    print("- Spooling Directory Source: 监控目录，处理新文件")
    print("  type = spooldir")
    print("  spoolDir = /var/log/flume/spool")
    print("  fileSuffix = .COMPLETED")
    print("- Kafka Source: 从 Kafka 主题读取数据")
    print("  type = org.apache.flume.source.kafka.KafkaSource")
    print("  kafka.bootstrap.servers = localhost:9092")
    print("  kafka.topics = logs")

# 6. Flume 常用 Channel
def flume_channels():
    print("\n6. Flume 常用 Channel:")
    print("- Memory Channel: 内存通道，速度快但易丢失数据")
    print("  type = memory")
    print("  capacity = 1000")
    print("  transactionCapacity = 100")
    print("- File Channel: 文件通道，持久化存储，可靠性高")
    print("  type = file")
    print("  checkpointDir = /var/lib/flume/checkpoint")
    print("  dataDirs = /var/lib/flume/data")
    print("  capacity = 1000000")
    print("- Kafka Channel: 使用 Kafka 作为通道")
    print("  type = org.apache.flume.channel.kafka.KafkaChannel")
    print("  kafka.bootstrap.servers = localhost:9092")
    print("  kafka.topic = flume-channel")

# 7. Flume 常用 Sink
def flume_sinks():
    print("\n7. Flume 常用 Sink:")
    print("- Logger Sink: 输出到控制台")
    print("  type = logger")
    print("- HDFS Sink: 输出到 HDFS")
    print("  type = hdfs")
    print("  hdfs.path = hdfs://localhost:9000/flume/logs")
    print("  hdfs.filePrefix = events-")
    print("  hdfs.fileSuffix = .log")
    print("  hdfs.rollInterval = 3600")
    print("  hdfs.rollSize = 134217728")
    print("  hdfs.rollCount = 0")
    print("- Kafka Sink: 输出到 Kafka")
    print("  type = org.apache.flume.sink.kafka.KafkaSink")
    print("  kafka.bootstrap.servers = localhost:9092")
    print("  kafka.topic = flume-output")
    print("- HBase Sink: 输出到 HBase")
    print("  type = org.apache.flume.sink.hbase.HBaseSink")
    print("  hbase.table = logs")
    print("  hbase.columnFamily = cf")

# 8. Flume 拦截器
def flume_interceptors():
    print("\n8. Flume 拦截器:")
    print("- 常用拦截器:")
    print("  - Timestamp Interceptor: 添加时间戳")
    print("  - Host Interceptor: 添加主机名")
    print("  - Static Interceptor: 添加静态值")
    print("  - Regex Filter Interceptor: 根据正则表达式过滤")
    print("  - Regex Extractor Interceptor: 从事件中提取数据")
    print("- 配置示例:")
    print("  agent.sources.r1.interceptors = i1 i2")
    print("  agent.sources.r1.interceptors.i1.type = timestamp")
    print("  agent.sources.r1.interceptors.i2.type = host")
    print("  agent.sources.r1.interceptors.i2.hostHeader = hostname")

# 9. Flume 拓扑结构
def flume_topologies():
    print("\n9. Flume 拓扑结构:")
    print("- 单 Agent 拓扑: 简单的数据源 -> 通道 -> 目的地")
    print("- 多 Agent 拓扑: 多个 Agent 串联，实现数据的多级处理")
    print("- 扇入拓扑: 多个 Source 向同一个 Channel 发送数据")
    print("- 扇出拓扑: 一个 Source 向多个 Channel 发送数据")
    print("- 配置示例 (多 Agent):")
    print("  # 第一级 Agent")
    print("  agent1.sources = r1")
    print("  agent1.sinks = k1")
    print("  agent1.channels = c1")
    print("  agent1.sinks.k1.type = avro")
    print("  agent1.sinks.k1.hostname = localhost")
    print("  agent1.sinks.k1.port = 41414")
    print("  ")
    print("  # 第二级 Agent")
    print("  agent2.sources = r1")
    print("  agent2.sinks = k1")
    print("  agent2.channels = c1")
    print("  agent2.sources.r1.type = avro")
    print("  agent2.sources.r1.bind = localhost")
    print("  agent2.sources.r1.port = 41414")

# 10. Flume 最佳实践
def flume_best_practices():
    print("\n10. Flume 最佳实践:")
    print("- 选择合适的 Channel:")
    print("  - 对可靠性要求高: File Channel")
    print("  - 对性能要求高: Memory Channel")
    print("- 合理配置缓冲区大小:")
    print("  - 根据内存情况调整 capacity 和 transactionCapacity")
    print("- 优化 HDFS Sink:")
    print("  - 合理设置 rollInterval、rollSize 和 rollCount")
    print("  - 使用压缩减少存储空间")
    print("- 监控和日志:")
    print("  - 监控 Agent 状态和性能")
    print("  - 定期检查日志文件")
    print("- 故障处理:")
    print("  - 配置多个 Sink 提高可靠性")
    print("  - 使用 File Channel 防止数据丢失")

if __name__ == "__main__":
    # 执行所有演示
    flume_basics()
    flume_architecture()
    flume_configuration()
    flume_start_stop()
    flume_sources()
    flume_channels()
    flume_sinks()
    flume_interceptors()
    flume_topologies()
    flume_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")