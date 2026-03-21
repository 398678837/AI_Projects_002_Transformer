#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kafka 消息队列演示

本脚本演示 Kafka 的基本概念、操作和使用方法。
"""

import os
import sys

print("Kafka 消息队列演示")
print("=" * 50)

# 1. Kafka 基本概念
def kafka_basics():
    print("\n1. Kafka 基本概念:")
    print("- Kafka 是一个分布式的流处理平台")
    print("- 设计用于高吞吐量、低延迟的消息传递")
    print("- 支持发布/订阅模式和流处理")
    print("- 适合处理实时数据管道和流分析")

# 2. Kafka 架构
def kafka_architecture():
    print("\n2. Kafka 架构:")
    print("- Broker: Kafka 服务器，存储消息")
    print("- Topic: 消息的分类，类似于队列")
    print("- Partition: 主题的分区，提高并行度")
    print("- Producer: 生产者，发送消息到 Kafka")
    print("- Consumer: 消费者，从 Kafka 读取消息")
    print("- Consumer Group: 消费者组，实现负载均衡和容错")
    print("- Zookeeper: 管理 Kafka 集群元数据")

# 3. Kafka 安装和配置
def kafka_installation():
    print("\n3. Kafka 安装和配置:")
    print("- 下载 Kafka: https://kafka.apache.org/downloads")
    print("- 解压安装包: tar -xzf kafka_2.13-3.5.0.tgz")
    print("- 配置环境变量:")
    print("  export KAFKA_HOME=/path/to/kafka")
    print("  export PATH=$PATH:$KAFKA_HOME/bin")
    print("- 主要配置文件:")
    print("  - server.properties: Kafka 服务器配置")
    print("  - zookeeper.properties: ZooKeeper 配置")

# 4. Kafka 启动和停止
def kafka_start_stop():
    print("\n4. Kafka 启动和停止:")
    print("- 启动 ZooKeeper:")
    print("  bin/zookeeper-server-start.sh config/zookeeper.properties")
    print("- 启动 Kafka 服务器:")
    print("  bin/kafka-server-start.sh config/server.properties")
    print("- 停止 Kafka 服务器:")
    print("  bin/kafka-server-stop.sh")
    print("- 停止 ZooKeeper:")
    print("  bin/zookeeper-server-stop.sh")

# 5. Kafka 命令行工具
def kafka_command_line():
    print("\n5. Kafka 命令行工具:")
    print("- 主题管理:")
    print("  - 创建主题: bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic test")
    print("  - 查看主题: bin/kafka-topics.sh --list --bootstrap-server localhost:9092")
    print("  - 查看主题详情: bin/kafka-topics.sh --describe --bootstrap-server localhost:9092 --topic test")
    print("  - 删除主题: bin/kafka-topics.sh --delete --bootstrap-server localhost:9092 --topic test")
    print("- 生产者:")
    print("  - 启动生产者: bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic test")
    print("- 消费者:")
    print("  - 启动消费者: bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning")

# 6. Kafka Python API
def kafka_python_api():
    print("\n6. Kafka Python API:")
    print("- 安装: pip install kafka-python")
    print("- 生产者示例:")
    print("  from kafka import KafkaProducer")
    print("  producer = KafkaProducer(bootstrap_servers=['localhost:9092'])")
    print("  producer.send('test', b'Hello, Kafka!')")
    print("  producer.flush()")
    print("  producer.close()")
    print("- 消费者示例:")
    print("  from kafka import KafkaConsumer")
    print("  consumer = KafkaConsumer('test', bootstrap_servers=['localhost:9092'], group_id='test-group')")
    print("  for message in consumer:")
    print("      print(message.value)")

# 7. Kafka 高级特性
def kafka_advanced_features():
    print("\n7. Kafka 高级特性:")
    print("- 消息压缩:")
    print("  - 支持 GZIP, Snappy, LZ4 等压缩格式")
    print("  - 配置: compression.type=snappy")
    print("- 消息过期:")
    print("  - 配置: log.retention.hours=24")
    print("- 事务:")
    print("  - 支持原子性生产和消费")
    print("  - 配置: transactional.id=producer-1")
    print("- 幂等性:")
    print("  - 防止重复生产消息")
    print("  - 配置: enable.idempotence=true")

# 8. Kafka 性能优化
def kafka_performance_optimization():
    print("\n8. Kafka 性能优化:")
    print("- 生产者优化:")
    print("  - 批量发送: batch.size=16384")
    print("  -  linger.ms=10")
    print("  - 压缩: compression.type=snappy")
    print("- 消费者优化:")
    print("  - 批量拉取: fetch.max.bytes=52428800")
    print("  - 自动提交: enable.auto.commit=true")
    print("  - 提交间隔: auto.commit.interval.ms=5000")
    print("- 服务器优化:")
    print("  - 内存: heap.size=4G")
    print("  - 日志刷新: log.flush.interval.messages=10000")
    print("  - 分区数: 根据集群规模调整")

# 9. Kafka 监控
def kafka_monitoring():
    print("\n9. Kafka 监控:")
    print("- 内置工具:")
    print("  - kafka-topics.sh: 查看主题状态")
    print("  - kafka-consumer-groups.sh: 查看消费者组状态")
    print("  - kafka-topics.sh --describe: 查看分区和副本状态")
    print("- 第三方工具:")
    print("  - Kafka Manager: Web 界面管理工具")
    print("  - Prometheus + Grafana: 监控和可视化")
    print("  - ELK Stack: 日志分析")

# 10. Kafka 常见问题
def kafka_common_issues():
    print("\n10. Kafka 常见问题:")
    print("- 消息丢失:")
    print("  - 生产者: 配置 acks=all")
    print("  - 消费者: 正确处理偏移量")
    print("- 消息重复:")
    print("  - 生产者: 启用幂等性")
    print("  - 消费者: 处理重复消息")
    print("- 性能问题:")
    print("  - 检查分区数")
    print("  - 检查网络和磁盘 I/O")
    print("  - 调整批处理大小")
    print("- 集群不稳定:")
    print("  - 检查 ZooKeeper 状态")
    print("  - 检查磁盘空间")
    print("  - 检查网络连接")

if __name__ == "__main__":
    # 执行所有演示
    kafka_basics()
    kafka_architecture()
    kafka_installation()
    kafka_start_stop()
    kafka_command_line()
    kafka_python_api()
    kafka_advanced_features()
    kafka_performance_optimization()
    kafka_monitoring()
    kafka_common_issues()
    
    print("\n" + "=" * 50)
    print("演示完成！")