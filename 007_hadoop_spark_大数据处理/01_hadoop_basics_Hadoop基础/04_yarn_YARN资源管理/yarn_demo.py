#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARN 资源管理演示

本脚本演示 YARN 的基本概念、架构和使用方法。
"""

import os
import sys

print("YARN 资源管理演示")
print("=" * 50)

# 1. YARN 基本概念
def yarn_basics():
    print("\n1. YARN 基本概念:")
    print("- YARN (Yet Another Resource Negotiator) 是 Hadoop 的资源管理系统")
    print("- 负责管理集群中的资源（CPU、内存）和调度作业")
    print("- 支持多种计算框架，如 MapReduce、Spark、Tez 等")
    print("- 设计目标: 提高集群利用率，支持多种计算框架")

# 2. YARN 架构
def yarn_architecture():
    print("\n2. YARN 架构:")
    print("- ResourceManager (RM): 集群资源管理和作业调度")
    print("  - 处理客户端请求")
    print("  - 分配资源给应用")
    print("  - 监控 NodeManager")
    print("- NodeManager (NM): 节点资源管理")
    print("  - 管理节点上的资源")
    print("  - 启动和监控容器")
    print("  - 向 ResourceManager 报告节点状态")
    print("- ApplicationMaster (AM): 应用管理")
    print("  - 为应用申请资源")
    print("  - 协调任务执行")
    print("  - 监控任务状态")
    print("- Container: 资源分配的基本单位")
    print("  - 包含 CPU 和内存资源")
    print("  - 运行任务的环境")

# 3. YARN 工作流程
def yarn_workflow():
    print("\n3. YARN 工作流程:")
    print("1. 客户端提交应用到 ResourceManager")
    print("2. ResourceManager 分配第一个容器给 ApplicationMaster")
    print("3. ApplicationMaster 启动并向 ResourceManager 注册")
    print("4. ApplicationMaster 向 ResourceManager 申请资源")
    print("5. ResourceManager 分配资源并通知 NodeManager")
    print("6. NodeManager 创建容器并启动任务")
    print("7. ApplicationMaster 监控任务执行")
    print("8. 任务完成后，ApplicationMaster 向 ResourceManager 注销")

# 4. YARN 配置
def yarn_configuration():
    print("\n4. YARN 配置:")
    print("- 主要配置文件: yarn-site.xml")
    print("- 关键配置参数:")
    print("  - yarn.resourcemanager.hostname: ResourceManager 主机名")
    print("  - yarn.nodemanager.resource.memory-mb: 节点内存总量")
    print("  - yarn.nodemanager.resource.cpu-vcores: 节点 CPU 核心数")
    print("  - yarn.scheduler.minimum-allocation-mb: 最小内存分配")
    print("  - yarn.scheduler.maximum-allocation-mb: 最大内存分配")
    print("  - yarn.scheduler.minimum-allocation-vcores: 最小 CPU 分配")
    print("  - yarn.scheduler.maximum-allocation-vcores: 最大 CPU 分配")

# 5. YARN 命令行工具
def yarn_commands():
    print("\n5. YARN 命令行工具:")
    print("- 应用管理:")
    print("  - 列出应用: yarn application -list")
    print("  - 查看应用状态: yarn application -status <app_id>")
    print("  - 杀死应用: yarn application -kill <app_id>")
    print("- 节点管理:")
    print("  - 列出节点: yarn node -list")
    print("  - 查看节点状态: yarn node -status <node_id>")
    print("- 队列管理:")
    print("  - 查看队列: yarn queue -status <queue_name>")
    print("- 日志查看:")
    print("  - 查看应用日志: yarn logs -applicationId <app_id>")
    print("  - 查看容器日志: yarn logs -containerId <container_id>")

# 6. YARN 调度器
def yarn_schedulers():
    print("\n6. YARN 调度器:")
    print("- FIFO 调度器:")
    print("  - 先进先出，简单但可能导致资源利用率低")
    print("- Capacity 调度器:")
    print("  - 为不同队列分配固定容量")
    print("  - 支持队列层级和资源共享")
    print("- Fair 调度器:")
    print("  - 公平分配资源")
    print("  - 支持资源抢占")
    print("  - 适合多用户环境")

# 7. YARN 队列配置
def yarn_queue_configuration():
    print("\n7. YARN 队列配置:")
    print("- 配置文件: capacity-scheduler.xml 或 fair-scheduler.xml")
    print("- Capacity 调度器配置示例:")
    print("  <property>")
    print("    <name>yarn.scheduler.capacity.root.queues</name>")
    print("    <value>default,production,development</value>")
    print("  </property>")
    print("  <property>")
    print("    <name>yarn.scheduler.capacity.root.default.capacity</name>")
    print("    <value>40</value>")
    print("  </property>")
    print("  <property>")
    print("    <name>yarn.scheduler.capacity.root.production.capacity</name>")
    print("    <value>40</value>")
    print("  </property>")
    print("  <property>")
    print("    <name>yarn.scheduler.capacity.root.development.capacity</name>")
    print("    <value>20</value>")
    print("  </property>")

# 8. YARN 高可用
def yarn_high_availability():
    print("\n8. YARN 高可用:")
    print("- 配置多个 ResourceManager (Active/Standby)")
    print("- 使用 Zookeeper 进行状态管理和故障转移")
    print("- 关键配置:")
    print("  <property>")
    print("    <name>yarn.resourcemanager.ha.enabled</name>")
    print("    <value>true</value>")
    print("  </property>")
    print("  <property>")
    print("    <name>yarn.resourcemanager.cluster-id</name>")
    print("    <value>cluster1</value>")
    print("  </property>")
    print("  <property>")
    print("    <name>yarn.resourcemanager.ha.rm-ids</name>")
    print("    <value>rm1,rm2</value>")
    print("  </property>")

# 9. YARN 监控
def yarn_monitoring():
    print("\n9. YARN 监控:")
    print("- Web UI:")
    print("  - ResourceManager: http://rm_host:8088")
    print("  - NodeManager: http://nm_host:8042")
    print("- 命令行监控:")
    print("  - yarn top: 实时查看应用资源使用情况")
    print("  - yarn application -list -appStates RUNNING: 查看运行中的应用")
    print("- 日志监控:")
    print("  - YARN 日志目录: $HADOOP_HOME/logs/")
    print("  - 应用日志: 通过 Web UI 或 yarn logs 命令查看")

# 10. YARN 最佳实践
def yarn_best_practices():
    print("\n10. YARN 最佳实践:")
    print("- 资源配置:")
    print("  - 根据节点硬件配置设置合理的资源总量")
    print("  - 为应用设置适当的资源请求")
    print("  - 避免过度分配资源")
    print("- 调度器选择:")
    print("  - 单用户环境: FIFO 调度器")
    print("  - 多用户环境: Capacity 或 Fair 调度器")
    print("  - 关键任务: 使用 Capacity 调度器并设置专用队列")
    print("- 监控和调优:")
    print("  - 定期监控集群资源使用情况")
    print("  - 根据应用特点调整配置")
    print("  - 及时清理失败的应用")

if __name__ == "__main__":
    # 执行所有演示
    yarn_basics()
    yarn_architecture()
    yarn_workflow()
    yarn_configuration()
    yarn_commands()
    yarn_schedulers()
    yarn_queue_configuration()
    yarn_high_availability()
    yarn_monitoring()
    yarn_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")