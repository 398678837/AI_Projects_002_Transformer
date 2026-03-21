#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphX 图计算演示

本脚本演示 Spark GraphX 的基本概念、操作和使用方法。
"""

import os
import sys

print("GraphX 图计算演示")
print("=" * 50)

# 1. GraphX 基本概念
def graphx_basics():
    print("\n1. GraphX 基本概念:")
    print("- GraphX 是 Spark 的图计算库")
    print("- 提供了分布式图处理能力")
    print("- 支持图的创建、转换和分析")
    print("- 提供了丰富的图算法")
    print("- 基于 RDD API")

# 2. 图的表示
def graph_representation():
    print("\n2. 图的表示:")
    print("- Graph[VD, ED]: 图由顶点和边组成")
    print("  - VD: 顶点属性类型")
    print("  - ED: 边属性类型")
    print("- 顶点: (vertexId, vertexAttribute)")
    print("- 边: (srcId, dstId, edgeAttribute)")

# 3. 图的创建
def graph_creation():
    print("\n3. 图的创建:")
    print("- 从 RDD 创建:")
    print("  from pyspark import SparkContext")
    print("  from graphframes import GraphFrame")
    print("  ")
    print("  sc = SparkContext('local', 'GraphCreation')")
    print("  spark = SparkSession.builder.appName('GraphCreation').getOrCreate()")
    print("  ")
    print("  # 创建顶点 DataFrame")
    print("  vertices = spark.createDataFrame([")
    print("      (1, 'Alice', 25),")
    print("      (2, 'Bob', 30),")
    print("      (3, 'Charlie', 35),")
    print("      (4, 'David', 40),")
    print("      (5, 'Eve', 45)")
    print("  ], ['id', 'name', 'age'])")
    print("  ")
    print("  # 创建边 DataFrame")
    print("  edges = spark.createDataFrame([")
    print("      (1, 2, 'friend'),")
    print("      (2, 3, 'friend'),")
    print("      (3, 4, 'friend'),")
    print("      (4, 5, 'friend'),")
    print("      (5, 1, 'friend')")
    print("  ], ['src', 'dst', 'relationship'])")
    print("  ")
    print("  # 创建图")
    print("  graph = GraphFrame(vertices, edges)")

# 4. 图的操作
def graph_operations():
    print("\n4. 图的操作:")
    print("- 查看顶点:")
    print("  graph.vertices.show()")
    print("- 查看边:")
    print("  graph.edges.show()")
    print("- 查看入度:")
    print("  graph.inDegrees.show()")
    print("- 查看出度:")
    print("  graph.outDegrees.show()")
    print("- 过滤顶点:")
    print("  graph.vertices.filter('age > 30').show()")
    print("- 过滤边:")
    print("  graph.edges.filter('relationship = "friend"').show()")

# 5. 图的转换
def graph_transformations():
    print("\n5. 图的转换:")
    print("- 顶点转换:")
    print("  # 增加顶点属性")
    print("  new_vertices = graph.vertices.withColumn('age_group', when(col('age') < 30, 'young').otherwise('old'))")
    print("  new_graph = GraphFrame(new_vertices, graph.edges)")
    print("- 边转换:")
    print("  # 增加边属性")
    print("  new_edges = graph.edges.withColumn('weight', lit(1.0))")
    print("  new_graph = GraphFrame(graph.vertices, new_edges)")
    print("- 子图:")
    print("  # 创建子图")
    print("  subgraph = graph.filterVertices('age > 30').filterEdges('relationship = "friend"')")

# 6. 图算法
def graph_algorithms():
    print("\n6. 图算法:")
    print("- PageRank:")
    print("  # 运行 PageRank 算法")
    print("  results = graph.pageRank(resetProbability=0.15, maxIter=10)")
    print("  results.vertices.select('id', 'name', 'pagerank').show()")
    print("- 连通组件:")
    print("  # 运行连通组件算法")
    print("  components = graph.connectedComponents()")
    print("  components.select('id', 'name', 'component').show()")
    print("- 三角形计数:")
    print("  # 运行三角形计数算法")
    print("  triangle_counts = graph.triangleCount()")
    print("  triangle_counts.select('id', 'name', 'count').show()")
    print("- 最短路径:")
    print("  # 运行最短路径算法")
    print("  shortest_paths = graph.shortestPaths(landmarks=[1, 5])")
    print("  shortest_paths.select('id', 'name', 'distances').show()")

# 7. 图遍历
def graph_traversal():
    print("\n7. 图遍历:")
    print("- BFS (广度优先搜索):")
    print("  # 运行 BFS 算法")
    print("  paths = graph.bfs(fromExpr='id = 1', toExpr='id = 5', maxPathLength=5)")
    print("  paths.show()")
    print("- 模式匹配:")
    print("  # 使用 Cypher 风格的模式匹配")
    print("  results = graph.find('(a)-[e]->(b); (b)-[e2]->(c)')")
    print("  results.show()")

# 8. 图的持久化
def graph_persistence():
    print("\n8. 图的持久化:")
    print("- 保存图:")
    print("  # 保存顶点")
    print("  graph.vertices.write.parquet('hdfs://path/to/vertices')")
    print("  # 保存边")
    print("  graph.edges.write.parquet('hdfs://path/to/edges')")
    print("- 加载图:")
    print("  # 加载顶点")
    print("  vertices = spark.read.parquet('hdfs://path/to/vertices')")
    print("  # 加载边")
    print("  edges = spark.read.parquet('hdfs://path/to/edges')")
    print("  # 创建图")
    print("  graph = GraphFrame(vertices, edges)")

# 9. 示例应用
def example_applications():
    print("\n9. 示例应用:")
    print("- 社交网络分析:")
    print("  # 分析社交网络中的好友关系")
    print("  # 1. 创建社交网络图")
    print("  # 2. 运行 PageRank 算法")
    print("  # 3. 分析影响力最大的用户")
    print("  ")
    print("  from pyspark.sql import SparkSession")
    print("  from graphframes import GraphFrame")
    print("  ")
    print("  spark = SparkSession.builder.appName('SocialNetworkAnalysis').getOrCreate()")
    print("  ")
    print("  # 创建顶点 DataFrame")
    print("  vertices = spark.createDataFrame([")
    print("      (1, 'Alice', 25),")
    print("      (2, 'Bob', 30),")
    print("      (3, 'Charlie', 35),")
    print("      (4, 'David', 40),")
    print("      (5, 'Eve', 45)")
    print("  ], ['id', 'name', 'age'])")
    print("  ")
    print("  # 创建边 DataFrame")
    print("  edges = spark.createDataFrame([")
    print("      (1, 2, 'friend'),")
    print("      (2, 3, 'friend'),")
    print("      (3, 4, 'friend'),")
    print("      (4, 5, 'friend'),")
    print("      (5, 1, 'friend'),")
    print("      (1, 3, 'friend'),")
    print("      (2, 4, 'friend')")
    print("  ], ['src', 'dst', 'relationship'])")
    print("  ")
    print("  # 创建图")
    print("  graph = GraphFrame(vertices, edges)")
    print("  ")
    print("  # 运行 PageRank")
    print("  results = graph.pageRank(resetProbability=0.15, maxIter=10)")
    print("  ")
    print("  # 显示 PageRank 结果")
    print("  results.vertices.select('id', 'name', 'pagerank').orderBy('pagerank', ascending=False).show()")

# 10. 最佳实践
def best_practices():
    print("\n10. 最佳实践:")
    print("- 数据准备:")
    print("  - 确保顶点 ID 唯一")
    print("  - 合理设置分区数")
    print("- 算法选择:")
    print("  - 根据问题选择合适的图算法")
    print("  - 调整算法参数以获得最佳性能")
    print("- 性能优化:")
    print("  - 使用缓存减少重复计算")
    print("  - 合理设置内存和核心数")
    print("- 错误处理:")
    print("  - 处理缺失数据")
    print("  - 处理无效边")
    print("- 结果分析:")
    print("  - 可视化图结构")
    print("  - 分析算法结果的含义")

if __name__ == "__main__":
    # 执行所有演示
    graphx_basics()
    graph_representation()
    graph_creation()
    graph_operations()
    graph_transformations()
    graph_algorithms()
    graph_traversal()
    graph_persistence()
    example_applications()
    best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")