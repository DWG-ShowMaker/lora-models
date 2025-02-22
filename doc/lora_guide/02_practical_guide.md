# LoRA实践指南

## 1. 环境准备

### 1.1 基础环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装基础依赖
pip install torch>=2.0.0
pip install transformers>=4.37.0
pip install peft>=0.7.0
pip install accelerate>=0.25.0
```

### 1.2 显存优化配置
```python
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

def setup_model_with_memory_optimization():
    # 创建加速器
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=4
    )
    
    # 加载模型时启用内存优化
    model = AutoModelForCausalLM.from_pretrained(
        "qwen/Qwen2.5-7B",
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False  # 训练时禁用KV缓存以节省显存
    )
    
    # 启用梯度检查点
    model.gradient_checkpointing_enable()
    
    return accelerator, model
```

## 2. 数据准备

### 2.1 数据格式化
```python
def format_conversation_data(conversations):
    """格式化对话数据"""
    formatted_data = []
    for conv in conversations:
        formatted = {
            "system": conv.get("system", "你是一个有帮助的AI助手"),
            "conversation": []
        }
        
        for turn in conv["dialogue"]:
            formatted["conversation"].append({
                "human": turn["user"],
                "assistant": turn["assistant"]
            })
        
        formatted_data.append(formatted)
    
    return formatted_data

# 使用示例
sample_data = [
    {
        "system": "你是一个医疗助手",
        "dialogue": [
            {"user": "头痛怎么办？", "assistant": "建议多休息，必要时服用止痛药"}
        ]
    }
]
formatted = format_conversation_data(sample_data)
```

### 2.2 数据集创建
```python
from datasets import Dataset
import json

def create_dataset(data_file, split_ratio=0.1):
    """创建训练集和验证集"""
    # 读取数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 创建数据集
    dataset = Dataset.from_dict({
        'system': [d['system'] for d in data],
        'conversation': [d['conversation'] for d in data]
    })
    
    # 分割数据集
    dataset = dataset.train_test_split(
        test_size=split_ratio,
        shuffle=True,
        seed=42
    )
    
    return dataset['train'], dataset['test']
```

## 3. 模型训练

### 3.1 LoRA配置
```python
from peft import LoraConfig, get_peft_model

def setup_lora_model(base_model):
    """配置LoRA模型"""
    # LoRA配置
    lora_config = LoraConfig(
        r=8,  # LoRA秩
        lora_alpha=32,  # LoRA alpha参数
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 创建PEFT模型
    model = get_peft_model(base_model, lora_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    return model
```

### 3.2 训练循环
```python
from transformers import Trainer, TrainingArguments
import wandb

def train_lora_model(model, tokenizer, train_dataset, eval_dataset):
    """训练LoRA模型"""
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True
    )
    
    # 初始化wandb
    wandb.init(project="lora-training")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            'input_ids': torch.stack([f['input_ids'] for f in data]),
            'attention_mask': torch.stack([f['attention_mask'] for f in data]),
            'labels': torch.stack([f['input_ids'] for f in data])
        }
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("final_model")
    wandb.finish()
```

## 4. 模型评估

### 4.1 评估指标
```python
def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    
    # 计算困惑度
    perplexity = torch.exp(
        torch.tensor(predictions.mean())
    ).item()
    
    # 计算准确率
    predictions = predictions.argmax(-1)
    mask = labels != -100
    accuracy = (predictions[mask] == labels[mask]).mean().item()
    
    return {
        "perplexity": perplexity,
        "accuracy": accuracy
    }
```

### 4.2 生成评估
```python
def evaluate_generation(model, tokenizer, test_prompts):
    """评估生成质量"""
    results = []
    
    for prompt in test_prompts:
        # 生成回复
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    return results
```

## 5. 模型部署

### 5.1 模型导出
```python
def export_lora_model(model, output_dir):
    """导出LoRA模型"""
    # 保存LoRA权重
    model.save_pretrained(output_dir)
    
    # 获取合并后的权重
    merged_model = model.merge_and_unload()
    
    # 保存完整模型（可选）
    merged_model.save_pretrained(
        f"{output_dir}/merged",
        safe_serialization=True
    )
    
    return output_dir
```

### 5.2 推理服务
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class GenerationResponse(BaseModel):
    response: str
    generation_time: float

@app.post("/generate")
async def generate(request: GenerationRequest):
    start_time = time.time()
    
    # 生成回复
    inputs = tokenizer(
        request.prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        do_sample=True
    )
    
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    
    generation_time = time.time() - start_time
    
    return GenerationResponse(
        response=response,
        generation_time=generation_time
    )
```

## 6. 性能优化

### 6.1 批处理推理
```python
def batch_inference(model, tokenizer, prompts, batch_size=4):
    """批量推理"""
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # 批量编码
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # 批量生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # 解码结果
        responses = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        
        results.extend(responses)
    
    return results
```

### 6.2 量化推理
```python
def setup_quantized_model():
    """设置量化模型"""
    from transformers import BitsAndBytesConfig
    
    # 4bit量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    
    # 加载量化模型
    model = AutoModelForCausalLM.from_pretrained(
        "qwen/Qwen2.5-7B",
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model
```

## 7. 监控和日志

### 7.1 训练监控
```python
def setup_monitoring():
    """设置训练监控"""
    import wandb
    from torch.utils.tensorboard import SummaryWriter
    
    # wandb配置
    wandb.init(
        project="lora-training",
        config={
            "model": "Qwen2.5-7B",
            "lora_rank": 8,
            "batch_size": 4,
            "learning_rate": 2e-4
        }
    )
    
    # tensorboard配置
    writer = SummaryWriter("runs/lora_training")
    
    return wandb, writer

def log_metrics(step, metrics, wandb, writer):
    """记录指标"""
    for name, value in metrics.items():
        wandb.log({name: value}, step=step)
        writer.add_scalar(name, value, step)
```

### 7.2 推理监控
```python
def setup_inference_monitoring():
    """设置推理监控"""
    import prometheus_client as prom
    
    # 定义指标
    LATENCY = prom.Histogram(
        'generation_latency_seconds',
        'Time spent processing generation requests',
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
    )
    
    REQUESTS = prom.Counter(
        'generation_requests_total',
        'Total number of generation requests'
    )
    
    ERROR_RATE = prom.Counter(
        'generation_errors_total',
        'Total number of generation errors'
    )
    
    return LATENCY, REQUESTS, ERROR_RATE
```

## 8. 错误处理

### 8.1 训练错误处理
```python
class TrainingError(Exception):
    """训练错误基类"""
    pass

class OutOfMemoryError(TrainingError):
    """显存不足错误"""
    pass

def handle_training_error(e):
    """处理训练错误"""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        raise OutOfMemoryError(
            "显存不足，建议：\n"
            "1. 减小batch_size\n"
            "2. 启用梯度检查点\n"
            "3. 使用混合精度训练"
        )
    elif isinstance(e, ValueError):
        if "nan" in str(e).lower():
            raise TrainingError(
                "梯度爆炸，建议：\n"
                "1. 降低学习率\n"
                "2. 增加梯度裁剪阈值"
            )
    else:
        raise TrainingError(f"未知错误: {str(e)}")
```

### 8.2 推理错误处理
```python
class InferenceError(Exception):
    """推理错误基类"""
    pass

def safe_generate(model, tokenizer, prompt, max_retries=3):
    """安全生成函数"""
    for attempt in range(max_retries):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if attempt == max_retries - 1:
                raise InferenceError("显存不足，无法完成生成")
            continue
            
        except Exception as e:
            raise InferenceError(f"生成错误: {str(e)}")
``` 