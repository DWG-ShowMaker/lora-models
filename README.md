# Qwen 2.5 LoRA 微调项目

本项目展示了如何使用LoRA技术对Qwen 2.5模型进行微调，实现特定问答对话。
并且使用VLLM进行高性能部署，支持批处理推理和动态批处理。

## 项目结构
```
.
├── README.md              # 项目说明文档
├── requirements.txt       # 项目依赖
├── config                # 配置文件目录
│   ├── lora_config.py    # LoRA训练配置
│   └── deploy_config.py  # 部署配置
├── src                   # 源代码目录
│   ├── prepare.py       # 数据和模型准备脚本
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   ├── deploy.py        # 部署脚本
│   └── utils.py         # 工具函数
├── data                  # 数据目录
│   ├── raw             # 原始数据
│   └── processed       # 处理后的数据
├── checkpoints          # 模型检查点目录
└── logs                 # 日志目录
```

## 环境要求
- Python 3.8+
- CUDA 11.7+
- 至少16GB显存的GPU
- 至少20GB磁盘空间（用于存储模型和数据）

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据和模型
```bash
python src/prepare.py
```
这个步骤会：
- 创建必要的目录结构
- 下载Qwen 2.5模型（约14GB）
- 创建示例训练数据集

你可以通过编辑 `src/prepare.py` 中的 `prepare_dataset` 函数来自定义训练数据。数据格式为JSONL，每行包含一个对话样本：
```json
{
    "conversations": [
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "助手回复"}
    ]
}
```

### 3. 训练模型
```bash
python src/train.py
```

训练配置可以在 `config/lora_config.py` 中修改，主要参数包括：
- LoRA配置（rank、alpha、dropout等）
- 训练参数（学习率、batch size、训练轮数等）
- 模型参数（模型路径、最大序列长度等）

### 4. 评估模型
```bash
python src/evaluate.py
```

评估脚本会：
- 加载训练好的LoRA模型
- 在测试集上进行推理
- 生成评估报告

### 5. 部署服务
```bash
python src/deploy.py
```

部署配置可以在 `config/deploy_config.py` 中修改，包括：
- 服务参数（host、port、workers）
- VLLM参数（tensor_parallel_size、gpu_memory_utilization等）
- 推理参数（max_new_tokens、temperature、top_p等）

## API使用

### 1. 文本生成接口
```bash
POST /generate
```
请求体：
```json
{
    "prompt": "你的输入文本",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "num_return_sequences": 1
}
```

### 2. 健康检查接口
```bash
GET /health
```

### Python调用示例
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "你好，请介绍一下你自己",
        "max_new_tokens": 512
    }
)
print(response.json())
```

## 训练技巧

### 1. 显存优化
如果遇到显存不足的问题，可以：
- 减小batch_size（在lora_config.py中设置）
- 启用梯度检查点
- 减少LoRA rank
- 使用8bit量化训练

### 2. 训练稳定性
如果训练不稳定，可以：
- 调整学习率（建议在1e-4到5e-4之间）
- 增加warm-up步数
- 调整gradient_accumulation_steps
- 检查数据质量和格式

### 3. 推理优化
部署时可以：
- 调整gpu_memory_utilization（在deploy_config.py中）
- 使用tensor_parallel进行多卡推理
- 开启KV缓存
- 使用批处理提高吞吐量

## 常见问题

1. 模型下载失败
   - 检查网络连接
   - 确保有足够的磁盘空间
   - 尝试使用镜像源

2. 训练中断
   - 检查显存使用情况
   - 查看日志文件定位错误
   - 确保数据格式正确

3. 部署问题
   - 检查端口占用情况
   - 确认GPU可用性
   - 查看服务日志

## 更新日志
- 2024-03-21: 项目初始化
- 2024-03-21: 添加数据准备脚本
- 2024-03-21: 添加VLLM部署支持

## 许可证
本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献
欢迎提交 Issue 和 Pull Request！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。 