#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapReduce 计算演示

本脚本演示 MapReduce 的基本概念、工作原理和使用方法。
"""

import os
import sys

print("MapReduce 计算演示")
print("=" * 50)

# 1. MapReduce 基本概念
def mapreduce_basics():
    print("\n1. MapReduce 基本概念:")
    print("- MapReduce 是一种分布式计算框架，用于处理大规模数据集")
    print("- 由 Google 提出，Hadoop 实现了开源版本")
    print("- 核心思想: 将计算分为 Map 和 Reduce 两个阶段")
    print("- 适合处理批处理任务，不适合实时处理")

# 2. MapReduce 工作原理
def mapreduce_working_principle():
    print("\n2. MapReduce 工作原理:")
    print("- Map 阶段:")
    print("  1. 输入数据被分割成多个 split")
    print("  2. 每个 split 由一个 Map 任务处理")
    print("  3. Map 函数将输入转换为 (key, value) 对")
    print("- Shuffle 阶段:")
    print("  1. 对 Map 输出进行排序和分组")
    print("  2. 将相同 key 的 value 汇总到一起")
    print("  3. 将结果分发到对应的 Reduce 任务")
    print("- Reduce 阶段:")
    print("  1. 对每个 key 的 value 集合进行处理")
    print("  2. 生成最终结果")
    print("  3. 将结果输出到 HDFS")

# 3. MapReduce 示例: 单词计数
def mapreduce_wordcount_example():
    print("\n3. MapReduce 示例: 单词计数")
    print("- Map 函数:")
    print("  def map(key, value):")
    print("      for word in value.split():")
    print("          emit(word, 1)")
    print("- Reduce 函数:")
    print("  def reduce(key, values):")
    print("      total = 0")
    print("      for v in values:")
    print("          total += v")
    print("      emit(key, total)")
    print("- 执行命令:")
    print("  hadoop jar $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-*.jar wordcount input output")

# 4. MapReduce 配置
def mapreduce_configuration():
    print("\n4. MapReduce 配置:")
    print("- 主要配置文件: mapred-site.xml")
    print("- 关键配置参数:")
    print("  - mapreduce.framework.name: 执行框架，默认为 yarn")
    print("  - mapreduce.map.memory.mb: Map 任务内存，默认为 1024MB")
    print("  - mapreduce.reduce.memory.mb: Reduce 任务内存，默认为 1024MB")
    print("  - mapreduce.map.cpu.vcores: Map 任务 CPU 核心数，默认为 1")
    print("  - mapreduce.reduce.cpu.vcores: Reduce 任务 CPU 核心数，默认为 1")
    print("  - mapreduce.job.reduces: Reduce 任务数量，默认为 1")

# 5. MapReduce Java API
def mapreduce_java_api():
    print("\n5. MapReduce Java API:")
    print("- 核心类:")
    print("  - Mapper: 实现 Map 功能")
    print("  - Reducer: 实现 Reduce 功能")
    print("  - Job: 配置和提交 MapReduce 作业")
    print("  - Configuration: 配置信息")
    print("- 示例代码结构:")
    print("  public class WordCount {")
    print("      public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {")
    print("          public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {")
    print("              // Map 逻辑")
    print("          }")
    print("      }")
    print("      public static class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {")
    print("          public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {")
    print("              // Reduce 逻辑")
    print("          }")
    print("      }")
    print("      public static void main(String[] args) throws Exception {")
    print("          // 作业配置和提交")
    print("      }")
    print("  }")

# 6. MapReduce 作业提交
def mapreduce_job_submission():
    print("\n6. MapReduce 作业提交:")
    print("- 命令行提交:")
    print("  hadoop jar jarfile.MainClass input output")
    print("- Java API 提交:")
    print("  Job job = Job.getInstance(conf, \"WordCount\");")
    print("  job.setJarByClass(WordCount.class);")
    print("  job.setMapperClass(Map.class);")
    print("  job.setReducerClass(Reduce.class);")
    print("  job.setOutputKeyClass(Text.class);")
    print("  job.setOutputValueClass(IntWritable.class);")
    print("  FileInputFormat.addInputPath(job, new Path(args[0]));")
    print("  FileOutputFormat.setOutputPath(job, new Path(args[1]));")
    print("  System.exit(job.waitForCompletion(true) ? 0 : 1);")

# 7. MapReduce 性能优化
def mapreduce_performance_optimization():
    print("\n7. MapReduce 性能优化:")
    print("- 输入数据优化:")
    print("  - 合理设置 split 大小")
    print("  - 使用 CombineFileInputFormat 处理小文件")
    print("- Map 阶段优化:")
    print("  - 使用 Combiner 减少网络传输")
    print("  - 合理设置 Map 任务内存和 CPU")
    print("- Shuffle 阶段优化:")
    print("  - 合理设置缓冲区大小")
    print("  - 使用压缩减少数据传输")
    print("- Reduce 阶段优化:")
    print("  - 合理设置 Reduce 任务数量")
    print("  - 避免数据倾斜")

# 8. MapReduce 数据倾斜
def mapreduce_data_skew():
    print("\n8. MapReduce 数据倾斜:")
    print("- 数据倾斜的原因:")
    print("  - 某些 key 的数据量远大于其他 key")
    print("  - 导致某些 Reduce 任务处理时间过长")
    print("- 解决方案:")
    print("  - 数据预处理，去除异常值")
    print("  - 使用随机前缀分散热点 key")
    print("  - 使用自定义 Partitioner 均匀分配数据")
    print("  - 增加 Reduce 任务数量")
    print("  - 使用 Combiner 减少数据量")

# 9. MapReduce 常见问题
def mapreduce_common_issues():
    print("\n9. MapReduce 常见问题:")
    print("- 作业失败:")
    print("  - 检查日志文件: yarn logs -applicationId <app_id>")
    print("  - 检查输入输出路径是否正确")
    print("  - 检查权限是否正确")
    print("- 内存溢出:")
    print("  - 增加任务内存配置")
    print("  - 优化代码，减少内存使用")
    print("- 执行速度慢:")
    print("  - 检查数据倾斜")
    print("  - 优化作业配置")
    print("  - 考虑使用 Spark 等更高效的计算框架")

# 10. MapReduce 最佳实践
def mapreduce_best_practices():
    print("\n10. MapReduce 最佳实践:")
    print("- 合理设计 Map 和 Reduce 函数:")
    print("  - 保持逻辑简单清晰")
    print("  - 避免在 Map 或 Reduce 中执行 heavy 操作")
    print("- 数据处理:")
    print("  - 预处理数据，去除无用信息")
    print("  - 使用适当的输入/输出格式")
    print("- 作业配置:")
    print("  - 根据数据量和集群资源设置合理的任务数量")
    print("  - 监控作业执行情况，及时调整配置")
    print("- 代码质量:")
    print("  - 编写健壮的代码，处理异常情况")
    print("  - 进行充分的测试")

if __name__ == "__main__":
    # 执行所有演示
    mapreduce_basics()
    mapreduce_working_principle()
    mapreduce_wordcount_example()
    mapreduce_configuration()
    mapreduce_java_api()
    mapreduce_job_submission()
    mapreduce_performance_optimization()
    mapreduce_data_skew()
    mapreduce_common_issues()
    mapreduce_best_practices()
    
    print("\n" + "=" * 50)
    print("演示完成！")