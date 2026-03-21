#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HBase 数据库演示

本脚本演示 HBase 的基本概念、操作和使用方法。
"""

import os
import sys

print("HBase 数据库演示")
print("=" * 50)

# 1. HBase 基本概念
def hbase_basics():
    print("\n1. HBase 基本概念:")
    print("- HBase 是基于 Hadoop 的分布式列存储数据库")
    print("- 设计用于存储大规模结构化数据")
    print("- 支持高并发读写操作")
    print("- 适合实时数据处理和随机读写")
    print("- 数据模型: 表、行、列族、列限定符、时间戳")

# 2. HBase 架构
def hbase_architecture():
    print("\n2. HBase 架构:")
    print("- HMaster: 管理集群，处理元数据操作")
    print("- RegionServer: 存储和管理数据，处理客户端请求")
    print("- ZooKeeper: 管理集群状态，协调 HMaster 选举")
    print("- HDFS: 存储 HBase 数据文件")
    print("- Region: 表的分区，每个 Region 包含一定范围的行")

# 3. HBase 数据模型
def hbase_data_model():
    print("\n3. HBase 数据模型:")
    print("- 表 (Table): 数据的集合")
    print("- 行 (Row): 由行键 (Row Key) 唯一标识")
    print("- 列族 (Column Family): 列的集合，物理上存储在一起")
    print("- 列限定符 (Column Qualifier): 列族内的具体列")
    print("- 单元格 (Cell): 由行键、列族、列限定符和时间戳唯一标识")
    print("- 时间戳 (Timestamp): 数据版本控制")

# 4. HBase 命令行操作
def hbase_shell_commands():
    print("\n4. HBase 命令行操作:")
    print("- 启动 HBase Shell:")
    print("  hbase shell")
    print("- 表操作:")
    print("  - 创建表: create 'users', 'info', 'address'")
    print("  - 查看表: list")
    print("  - 查看表结构: describe 'users'")
    print("  - 禁用表: disable 'users'")
    print("  - 启用表: enable 'users'")
    print("  - 删除表: drop 'users'")
    print("- 数据操作:")
    print("  - 插入数据: put 'users', 'user1', 'info:name', 'John'")
    print("  - 查询数据: get 'users', 'user1'")
    print("  - 扫描数据: scan 'users'")
    print("  - 删除数据: delete 'users', 'user1', 'info:name'")
    print("  - 删除整行: deleteall 'users', 'user1'")

# 5. HBase Java API
def hbase_java_api():
    print("\n5. HBase Java API:")
    print("- 核心类:")
    print("  - Configuration: 配置信息")
    print("  - Connection: 与 HBase 集群的连接")
    print("  - Table: 表操作")
    print("  - Admin: 管理操作")
    print("  - Put: 插入数据")
    print("  - Get: 查询数据")
    print("  - Scan: 扫描数据")
    print("  - Delete: 删除数据")
    print("- 示例代码:")
    print("  Configuration conf = HBaseConfiguration.create();")
    print("  try (Connection connection = ConnectionFactory.createConnection(conf);")
    print("       Admin admin = connection.getAdmin()) {")
    print("      // 创建表")
    print("      TableName tableName = TableName.valueOf(\"users\");")
    print("      TableDescriptorBuilder tableBuilder = TableDescriptorBuilder.newBuilder(tableName);")
    print("      ColumnFamilyDescriptor family = ColumnFamilyDescriptorBuilder.newBuilder(Bytes.toBytes(\"info\")).build();")
    print("      tableBuilder.setColumnFamily(family);")
    print("      admin.createTable(tableBuilder.build());")
    print("  }")

# 6. HBase Python API (happybase)
def hbase_python_api():
    print("\n6. HBase Python API (happybase):")
    print("- 安装: pip install happybase")
    print("- 示例代码:")
    print("  import happybase")
    print("  # 连接 HBase")
    print("  connection = happybase.Connection('localhost')")
    print("  # 创建表")
    print("  connection.create_table(")
    print("      'users',")
    print("      {'info': dict(), 'address': dict()}")
    print("  )")
    print("  # 获取表")
    print("  table = connection.table('users')")
    print("  # 插入数据")
    print("  table.put('user1', {'info:name': 'John', 'info:age': '30'})")
    print("  # 查询数据")
    print("  print(table.row('user1'))")
    print("  # 扫描数据")
    print("  for key, data in table.scan():")
    print("      print(key, data)")

# 7. HBase 性能优化
def hbase_performance_optimization():
    print("\n7. HBase 性能优化:")
    print("- 表设计优化:")
    print("  - 合理设计行键，避免热点问题")
    print("  - 合理设置列族数量，建议不超过 3 个")
    print("  - 为列族设置合适的压缩方式")
    print("- 配置优化:")
    print("  - 调整 RegionServer 内存分配")
    print("  - 设置合适的 Region 大小")
    print("  - 启用 Bloom Filter 提高查询性能")
    print("- 数据操作优化:")
    print("  - 使用批量操作减少网络往返")
    print("  - 合理使用缓存")
    print("  - 避免全表扫描")

# 8. HBase 高可用
def hbase_high_availability():
    print("\n8. HBase 高可用:")
    print("- 多 HMaster 配置:")
    print("  - 配置多个 HMaster，一个 Active，其他 Standby")
    print("  - 使用 ZooKeeper 进行故障转移")
    print("- RegionServer 高可用:")
    print("  - 数据自动复制到多个 RegionServer")
    print("  - 故障时自动重分配 Region")
    print("- ZooKeeper 高可用:")
    print("  - 配置 ZooKeeper 集群")

# 9. HBase 监控
def hbase_monitoring():
    print("\n9. HBase 监控:")
    print("- Web UI:")
    print("  - HMaster: http://hmaster:16010")
    print("  - RegionServer: http://regionserver:16030")
    print("- 命令行工具:")
    print("  - hbase hbck: 检查 HBase 一致性")
    print("  - hbase shell: 执行管理命令")
    print("- 日志文件:")
    print("  - HMaster 日志: $HBASE_HOME/logs/")
    print("  - RegionServer 日志: $HBASE_HOME/logs/")

# 10. HBase 常见问题
def hbase_common_issues():
    print("\n10. HBase 常见问题:")
    print("- 启动失败:")
    print("  - 检查 ZooKeeper 连接")
    print("  - 检查 HDFS 连接")
    print("  - 检查配置文件")
    print("- 性能问题:")
    print("  - 检查 Region 分布")
    print("  - 检查内存使用")
    print("  - 检查磁盘 I/O")
    print("- 数据一致性问题:")
    print("  - 运行 hbase hbck 检查")
    print("  - 修复不一致性")

if __name__ == "__main__":
    # 执行所有演示
    hbase_basics()
    hbase_architecture()
    hbase_data_model()
    hbase_shell_commands()
    hbase_java_api()
    hbase_python_api()
    hbase_performance_optimization()
    hbase_high_availability()
    hbase_monitoring()
    hbase_common_issues()
    
    print("\n" + "=" * 50)
    print("演示完成！")