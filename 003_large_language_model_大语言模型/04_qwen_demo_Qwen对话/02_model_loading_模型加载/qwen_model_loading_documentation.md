# Qwen模型加载详细文档

## 1. 加载方式

### 1.1 基本加载

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1.8B-Chat")
```

### 1.2 量化加载

- 4bit量化
- 8bit量化
- 降低显存需求

## 2. 版本选择

| 版本 | 参数量 | 显存需求 |
|------|--------|----------|
| 0.5B | 0.5B | 1GB |
| 1.8B | 1.8B | 4GB |
| 7B | 7B | 14GB |

## 3. 总结

根据硬件选择合适的模型版本。
