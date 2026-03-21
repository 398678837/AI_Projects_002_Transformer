#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hadoop 概述演示

本脚本演示 Hadoop 的基本概念、架构和使用方法。
"""

import os
import sys

print("Hadoop 概述演示")
print("=" * 50)

# 1. Hadoop 基本概念
def hadoop_basics():
    print("\n1. Hadoop 基本概念:")
    print("- Hadoop 是一个开源的分布式计算框架，用于处理大规模数据集")
    print("- 由 Apache 基金会开发和维护")
    print("- 主要组件包括 HDFS、MapReduce 和 YARN")
    print("- 设计理念: 移动计算比移动数据更高效")

# 2. Hadoop 架构
def hadoop_architecture():
    print("\n2. Hadoop 架构:")
    print("- 主从架构 (Master-Slave Architecture)")
    print("- 主节点: NameNode (HDFS), ResourceManager (YARN)")
    print("- 从节点: DataNode (HDFS), NodeManager (YARN)")
    print("- 高可靠性: 数据多副本存储")
    print("- 高扩展性: 可以线性扩展集群规模")

# 3. Hadoop 生态系统
def hadoop_ecosystem():
    print("\n3. Hadoop 生态系统:")
    print("- HDFS: 分布式文件系统")
    print("- MapReduce: 分布式计算框架")
    print("- YARN: 资源管理系统")
    print("- Hive: 数据仓库工具")
    print("- HBase: 分布式数据库")
    print("- Spark: 快速计算引擎")
    print("- Flume: 日志收集工具")
    print("- Kafka: 消息队列系统")

# 4. Hadoop 安装和配置
def hadoop_installation():
    print("\n4. Hadoop 安装和配置:")
    print("- 前提条件: Java JDK 1.8 或更高版本")
    print("- 下载 Hadoop 安装包: https://hadoop.apache.org/releases.html")
    print("- 配置环境变量:")
    print("  export HADOOP_HOME=/path/to/hadoop")
    print("  export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin")
    print("- 配置文件:")
    print("  - core-site.xml: 核心配置")
    print("  - hdfs-site.xml: HDFS 配置")
    print("  - yarn-site.xml: YARN 配置")
    print("  - mapred-site.xml: MapReduce 配置")

# 5. Hadoop 集群启动和停止
def hadoop_cluster_management():
    print("\n5. Hadoop 集群启动和停止:")
    print("- 启动 HDFS:")
    print("  start-dfs.sh")
    print("- 启动 YARN:")
    print("  start-yarn.sh")
    print("- 启动所有服务:")
    print("  start-all.sh")
    print("- 停止 HDFS:")
    print("  stop-dfs.sh")
    print("- 停止 YARN:")
    print("  stop-yarn.sh")
    print("- 停止所有服务:")
    print("  stop-all.sh")

# 6. Hadoop 命令行工具
def hadoop_command_line():
    print("\n6. Hadoop 命令行工具:")
    print("- HDFS 命令:")
    print("  - 创建目录: hdfs dfs -mkdir /path")
    print("  - 上传文件: hdfs dfs -put localfile /hdfs/path")
    print("  - 下载文件: hdfs dfs -get /hdfs/path localfile")
    print("  - 查看文件: hdfs dfs -cat /hdfs/path")
    print("  - 列出目录: hdfs dfs -ls /path")
    print("- MapReduce 命令:")
    print("  - 运行作业: hadoop jar jarfile.MainClass input output")

# 7. Hadoop 示例应用
def hadoop_examples():
    print("\n7. Hadoop 示例应用:")
    print("- 单词计数 (WordCount):")
    print("  输入: 文本文件")
    print("  输出: 每个单词出现的次数")
    print("  命令: hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount input output")
    print("- 排序 (Sort):")
    print("  输入: 键值对")
    print("  输出: 排序后的键值对")
    print("  命令: hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar sort input output")

# 8. Hadoop Web UI
def hadoop_web_ui():
    print("\n8. Hadoop Web UI:")
    print("- HDFS Web UI: http://namenode:9870")
    print("- YARN Web UI: http://resourcemanager:8088")
    print("- MapReduce JobHistory: http://jobhistoryserver:19888")

# 9. Hadoop 最佳实践
def hadoop_best_practices():
    print("\n9. Hadoop 最佳实践:")
    print("- 数据本地化: 将计算任务调度到数据所在节点")
    print("- 合理设置块大小: 默认为 128MB")
    print("- 合理设置副本数: 默认为 3")
    print("- 压缩数据: 减少网络传输和存储开销")
    print("- 使用合适的 InputFormat 和 OutputFormat")

# 10. Hadoop 常见问题
def hadoop_common_issues():
    print("\n10. Hadoop 常见问题:")
    print("- 权限问题: 确保用户有适当的权限")
    print("- 内存不足: 调整 JVM 堆大小")
    print("- 网络问题: 确保集群网络畅通")
    print("- 磁盘空间不足: 监控和清理磁盘空间")
    print("- 配置错误: 检查配置文件是否正确")

if __name__ == "__main__":
    # 执行所有演示
    hadoop_basics()
    hadoop_architecture()
    hadoop_ecosystem()
    hadoop_installation()
    hadoop_cluster_management()
    hadoop_command_line()
    hadoop_examples()
    hadoop_web_ui()
    hadoop_best_practices()
    hadoop_common_issues()
    
    print("\n" + "=" * 50)
    print("演示完成！")