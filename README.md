# Qwen 2.5 LoRA 微调项目

本项目使用LoRA技术对Qwen 2.5模型进行微调，以实现特定任务的优化。

## 项目结构
```
.
├── README.md              # 项目说明文档
├── requirements.txt       # 项目依赖
├── config                # 配置文件目录
│   └── lora_config.py    # LoRA训练配置
├── src                   # 源代码目录
│   ├── train.py         # 训练脚本
│   ├── evaluate.py      # 评估脚本
│   └── utils.py         # 工具函数
└── data                  # 数据目录
    └── processed        # 处理后的数据
```

## 环境要求
- Python 3.8+
- CUDA 11.7+
- 至少16GB显存的GPU

## 安装依赖
```bash
pip install -r requirements.txt
```

## 数据集
项目使用ModelScope的Muice-Dataset数据集：
- 训练集：`Moemuu/Muice-Dataset` (subset_name='default', split='train')
- 测试集：`Moemuu/Muice-Dataset` (subset_name='default', split='test')

## 训练方法
1. 配置训练参数
   编辑 `config/lora_config.py` 文件，设置LoRA参数和训练超参数

2. 开始训练
```bash
python src/train.py
```

3. 评估模型
```bash
python src/evaluate.py
```

## LoRA配置说明
- rank: 8
- alpha: 32
- dropout: 0.1
- target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']

## 注意事项
1. 确保有足够的GPU显存
2. 建议使用梯度检查点以减少显存使用
3. 可以根据实际情况调整batch_size和learning_rate

## 监控和日志
- 训练日志保存在 `logs` 目录
- 使用tensorboard监控训练过程
- 模型检查点保存在 `checkpoints` 目录

## 常见问题解决
1. 显存不足：
   - 减小batch_size
   - 启用梯度检查点
   - 减少LoRA rank

2. 训练不稳定：
   - 调整学习率
   - 检查数据预处理
   - 调整warm-up步数

## 更新日志
- 2024-03-21: 项目初始化 