#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDFS 分布式存储演示

本脚本演示 HDFS 的基本概念、操作和使用方法。
"""

import os
import sys

print("HDFS 分布式存储演示")
print("=" * 50)

# 1. HDFS 基本概念
def hdfs_basics():
    print("\n1. HDFS 基本概念:")
    print("- HDFS (Hadoop Distributed File System) 是 Hadoop 的分布式文件系统")
    print("- 设计用于存储大规模数据集")
    print("- 采用主从架构: NameNode (主节点) 和 DataNode (从节点)")
    print("- 数据以块 (Block) 为单位存储，默认为 128MB")
    print("- 数据多副本存储，默认为 3 个副本")

# 2. HDFS 架构
def hdfs_architecture():
    print("\n2. HDFS 架构:")
    print("- NameNode: 管理文件系统命名空间，维护文件和目录的元数据")
    print("- DataNode: 存储实际数据块，处理客户端的读写请求")
    print("- SecondaryNameNode: 辅助 NameNode 进行 checkpoint，提高可靠性")
    print("- Block: 数据存储的基本单位，默认为 128MB")
    print("- Replica: 数据块的副本，默认为 3 个")

# 3. HDFS 配置
def hdfs_configuration():
    print("\n3. HDFS 配置:")
    print("- 主要配置文件: hdfs-site.xml")
    print("- 核心配置: core-site.xml")
    print("- 关键配置参数:")
    print("  - dfs.replication: 副本数，默认为 3")
    print("  - dfs.blocksize: 块大小，默认为 134217728 (128MB)")
    print("  - dfs.namenode.name.dir: NameNode 元数据存储目录")
    print("  - dfs.datanode.data.dir: DataNode 数据存储目录")

# 4. HDFS 命令行操作
def hdfs_commands():
    print("\n4. HDFS 命令行操作:")
    print("- 目录操作:")
    print("  - 创建目录: hdfs dfs -mkdir /path")
    print("  - 创建多级目录: hdfs dfs -mkdir -p /path/to/dir")
    print("  - 列出目录: hdfs dfs -ls /path")
    print("  - 递归列出目录: hdfs dfs -ls -R /path")
    print("- 文件操作:")
    print("  - 上传文件: hdfs dfs -put localfile /hdfs/path")
    print("  - 下载文件: hdfs dfs -get /hdfs/path localfile")
    print("  - 查看文件: hdfs dfs -cat /hdfs/path")
    print("  - 复制文件: hdfs dfs -cp /hdfs/source /hdfs/destination")
    print("  - 移动文件: hdfs dfs -mv /hdfs/source /hdfs/destination")
    print("  - 删除文件: hdfs dfs -rm /hdfs/path")
    print("  - 删除目录: hdfs dfs -rm -r /hdfs/path")
    print("- 其他操作:")
    print("  - 查看文件大小: hdfs dfs -du -h /hdfs/path")
    print("  - 检查文件状态: hdfs dfs -stat /hdfs/path")
    print("  - 更改权限: hdfs dfs -chmod 755 /hdfs/path")
    print("  - 更改所有者: hdfs dfs -chown user:group /hdfs/path")

# 5. HDFS Java API
def hdfs_java_api():
    print("\n5. HDFS Java API:")
    print("- 核心类:")
    print("  - FileSystem: 文件系统操作的抽象类")
    print("  - Path: 表示 HDFS 中的路径")
    print("  - FSDataInputStream: 读取文件")
    print("  - FSDataOutputStream: 写入文件")
    print("- 示例代码:")
    print("  Configuration conf = new Configuration();")
    print("  FileSystem fs = FileSystem.get(conf);")
    print("  Path path = new Path(\"/hdfs/path\");")
    print("  if (fs.exists(path)) {")
    print("      // 文件存在")
    print("  }")

# 6. HDFS Python API (hdfs3)
def hdfs_python_api():
    print("\n6. HDFS Python API (hdfs3):")
    print("- 安装: pip install hdfs3")
    print("- 示例代码:")
    print("  from hdfs3 import HDFileSystem")
    print("  hdfs = HDFileSystem(host='namenode', port=9000)")
    print("  # 列出目录")
    print("  print(hdfs.ls('/'))")
    print("  # 上传文件")
    print("  hdfs.put('localfile', '/hdfs/path')")
    print("  # 下载文件")
    print("  hdfs.get('/hdfs/path', 'localfile')")

# 7. HDFS 数据均衡
def hdfs_balancing():
    print("\n7. HDFS 数据均衡:")
    print("- 数据均衡工具: hdfs dfsadmin -balancer")
    print("- 启动均衡器:")
    print("  hdfs dfsadmin -balancer")
    print("- 设置均衡阈值:")
    print("  hdfs dfsadmin -balancer -threshold 10")
    print("- 取消均衡:")
    print("  hdfs dfsadmin -balancer -cancel")

# 8. HDFS 安全模式
def hdfs_safe_mode():
    print("\n8. HDFS 安全模式:")
    print("- 安全模式是 HDFS 的一种特殊状态，只允许读操作")
    print("- 检查安全模式状态:")
    print("  hdfs dfsadmin -safemode get")
    print("- 进入安全模式:")
    print("  hdfs dfsadmin -safemode enter")
    print("- 离开安全模式:")
    print("  hdfs dfsadmin -safemode leave")
    print("- 等待安全模式:")
    print("  hdfs dfsadmin -safemode wait")

# 9. HDFS 故障处理
def hdfs_troubleshooting():
    print("\n9. HDFS 故障处理:")
    print("- NameNode 故障:")
    print("  - 使用 SecondaryNameNode 的 checkpoint 恢复")
    print("  - 配置 NameNode 高可用 (HA)")
    print("- DataNode 故障:")
    print("  - 自动复制数据到其他节点")
    print("  - 检查节点状态: hdfs dfsadmin -report")
    print("- 数据块损坏:")
    print("  - 自动检测和恢复")
    print("  - 手动检查: hdfs fsck /path")

# 10. HDFS 最佳实践
def hdfs_best_practices():
    print("\n10. HDFS 最佳实践:")
    print("- 合理设置块大小:")
    print("  - 大文件: 128MB 或 256MB")
    print("  - 小文件: 考虑使用 SequenceFile 或 Avro")
    print("- 合理设置副本数:")
    print("  - 关键数据: 3 个或更多副本")
    print("  - 非关键数据: 1-2 个副本")
    print("- 避免小文件:")
    print("  - 小文件会增加 NameNode 内存开销")
    print("  - 使用 HAR (Hadoop Archive) 或 SequenceFile 合并小文件")
    print("- 数据压缩:")
    print("  - 减少存储空间和网络传输")
    print("  - 常用压缩格式: Gzip, Snappy, LZO")

if __name__ == "__main__":
    # 执行所有演示
    hdfs_basics()
    hdfs_architecture()
    hdfs_configuration()
    hdfs_commands()
    hdfs_java_api()
    hdfs_python_api()
    hdfs_balancing()
    hdfs_safe_mode()
    hdfs_troubleshooting()
    hdfs_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")