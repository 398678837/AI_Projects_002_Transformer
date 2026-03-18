# 分词器教材

## 第一章：分词器概念

### 1.1 什么是分词器

分词器(Tokenization)将文本转换为模型可处理的token序列，是NLP处理的第一步。

### 1.2 分词方法

1. **Word (词级分词)**
   - 将文本按词分割
   - 词表大，稀疏
   - 例子：["I", "love", "learning"]

2. **Subword (子词分词)**
   - 将词分解为子词单元
   - 平衡词表大小和稀疏性
   - 例子：["I", "lov", "e", "learn", "ing"]

3. **Character (字符级分词)**
   - 将文本按字符分割
   - 词表小，冗余
   - 例子：["I", "l", "o", "v", "e"]

### 1.3 分词器的重要性

1. **影响模型性能**
   - 分词质量影响模型理解
   - 合适的分词方法提升性能

2. **影响计算效率**
   - 分词长度影响计算时间
   - 合适的分词方法提升效率

3. **影响模型大小**
   - 词表大小影响模型参数
   - 合适的分词方法减小模型

## 第二章：常用算法

### 2.1 BPE (Byte Pair Encoding)

**特点**：
- 迭代合并常见字节对
- 动态词表
- 子词分词

**优点**：
- 词表大小可控
- 处理未知词
- 训练效率高

**缺点**：
- 需要训练
- 可能分割不合理

**应用**：
- GPT系列
- RoBERTa

### 2.2 WordPiece

**特点**：
- 基于词频合并
- 子词分词
- 保持词根

**优点**：
- 保持词根
- 词表大小可控
- 处理未知词

**缺点**：
- 需要训练
- 可能分割不合理

**应用**：
- BERT
- T5

### 2.3 SentencePiece

**特点**：
- 原始文本训练
- 子词分词
- 语言无关

**优点**：
- 语言无关
- 词表大小可控
- 处理未知词

**缺点**：
- 需要训练
- 可能分割不合理

**应用**：
- Llama
- M2M-100

### 2.4 其他算法

**Unigram**：
- 基于概率模型
- 子词分词
- 语言模型

**Character-level**：
- 字符级分词
- 词表小
- 冗余高

## 第三章：使用方法

### 3.1 基本使用

```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 分词
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)

# 编码
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
```

### 3.2 高级使用

```python
from transformers import AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 分词
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# 编码
inputs = tokenizer(text, return_tensors="pt")
print("Input IDs:", inputs["input_ids"])
print("Attention Mask:", inputs["attention_mask"])

# 解码
decoded = tokenizer.decode(inputs["input_ids"][0])
print("Decoded:", decoded)
```

### 3.3 自定义分词器

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# 创建分词器
tokenizer = Tokenizer(models.BPE())

# 预分词
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# 解码器
tokenizer.decoder = decoders.ByteLevel()

# 训练
trainer = trainers.BpeTrainer(vocab_size=10000, min_frequency=2)
tokenizer.train(files=["path/to/text.txt"], trainer=trainer)

# 保存
tokenizer.save("tokenizer.json")
```

## 第四章：应用场景

### 4.1 文本分类

- 情感分析
- 主题分类
- 垃圾邮件检测

### 4.2 命名实体识别

- 人名识别
- 地名识别
- 组织识别

### 4.3 问答系统

- 开放域问答
- 机器阅读理解
- 对话系统

### 4.4 文本生成

- 文本续写
- 文本摘要
- 机器翻译

### 4.5 代码生成

- 代码补全
- 代码生成
- 代码翻译

## 第五章：最新进展

### 5.1 新算法

- 更高效的分词算法
- 更好的子词分词
- 语言特定分词器

### 5.2 性能优化

- 更快的分词速度
- 更小的内存占用
- 更好的并行处理

### 5.3 新功能

- 多语言分词器
- 特定任务分词器
- 自适应分词器

## 第六章：实践项目

### 6.1 分词器使用项目

1. 使用BERT分词器
2. 使用GPT分词器
3. 使用T5分词器

### 6.2 自定义分词器项目

1. 训练BPE分词器
2. 训练WordPiece分词器
3. 训练SentencePiece分词器

## 第七章：最佳实践

### 7.1 分词器选择

- 根据模型选择分词器
- 考虑语言特性
- 考虑任务需求

### 7.2 分词器使用

- 正确加载分词器
- 正确编码文本
- 正确解码输出

### 7.3 分词器优化

- 优化词表大小
- 优化分词速度
- 优化内存占用

## 第八章：习题

### 8.1 选择题

1. 以下哪个不是分词方法？
   - A) Word
   - B) Subword
   - C) Sentence
   - D) Character

2. 以下哪个算法不是常用的分词算法？
   - A) BPE
   - B) WordPiece
   - C) GPT
   - D) SentencePiece

3. 分词器对模型的影响不包括？
   - A) 性能
   - B) 效率
   - C) 颜色
   - D) 大小

### 8.2 填空题

1. 分词器是________处理的第一步。
2. BPE的全称是________________________。
3. WordPiece常用于________________模型。

### 8.3 简答题

1. 简述分词方法的种类。
2. 简述BPE算法的原理。
3. 简述分词器对模型的影响。

### 8.4 编程题

1. 使用BERT分词器处理文本。
2. 使用GPT分词器处理文本。
3. 训练自定义分词器。

## 第九章：总结

### 9.1 知识回顾

1. 分词器概念
2. 分词方法
3. 常用算法
4. 使用方法
5. 应用场景
6. 最新进展

### 9.2 学习建议

1. 理解概念
2. 动手实践
3. 阅读文档
4. 参与社区

### 9.3 进阶学习

1. 研究源码
2. 参与开源
3. 发表论文