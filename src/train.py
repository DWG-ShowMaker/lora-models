import os
import sys
import logging
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
from datetime import datetime
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.lora_config import config
from utils import preprocess_function, compute_metrics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory_exists():
    """确保必要的目录结构存在"""
    directories = [
        "data/raw",
        "data/processed",
        "checkpoints",
        "logs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def check_dataset_files():
    """检查数据集文件是否存在"""
    required_files = {
        "train": "data/processed/train.jsonl",
        "test": "data/processed/test.jsonl"
    }
    
    missing_files = []
    for name, file_path in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required dataset files: {', '.join(missing_files)}. "
            "Please run prepare.py first to create the datasets."
        )
    else:
        logger.info("All required dataset files found.")
        for name, file_path in required_files.items():
            file_size = os.path.getsize(file_path)
            logger.info(f"{name} dataset size: {file_size/1024:.2f} KB")

def check_model_files(model_dir):
    """检查模型文件是否完整"""
    required_files = ['config.json', 'tokenizer.json']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    return missing_files

def download_model(cache_dir='checkpoints'):
    """从ModelScope下载模型"""
    logger.info("Downloading model from ModelScope...")
    try:
        model_dir = snapshot_download(
            'qwen/Qwen2.5-7B',
            cache_dir=cache_dir,
            revision='master'
        )
        logger.info(f"Model downloaded successfully to {model_dir}")
        return model_dir
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {str(e)}")

def main():
    # 确保目录结构存在
    ensure_directory_exists()
    
    # 检查数据集文件
    check_dataset_files()
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 检查本地模型
    model_dir = os.path.join("checkpoints", "qwen/Qwen2.5-7B")
    if os.path.exists(model_dir):
        logger.info(f"Found local model at {model_dir}")
        missing_files = check_model_files(model_dir)
        if missing_files:
            logger.warning(f"Local model is incomplete. Missing files: {', '.join(missing_files)}")
            logger.info("Will download model from ModelScope...")
            model_dir = download_model()
    else:
        logger.info("Local model not found")
        model_dir = download_model()
    
    # 再次验证模型文件完整性
    missing_files = check_model_files(model_dir)
    if missing_files:
        raise FileNotFoundError(f"Model directory is still incomplete after download. Missing files: {', '.join(missing_files)}")
    
    # 加载tokenizer和模型
    logger.info("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        logger.info("Tokenizer loaded successfully")
        
        # 使用4bit量化加载模型
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={0: "12GB"}  # 限制GPU显存使用
        )
        
        # 启用梯度检查点和其他优化
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        # 清理缓存
        torch.cuda.empty_cache()
        
        logger.info(f"Model loaded successfully. Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")
    
    # 配置LoRA
    logger.info("Configuring LoRA")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # 减小LoRA秩
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 只在部分模块上应用LoRA
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    logger.info("LoRA configuration completed")
    
    # 加载数据集
    logger.info("Loading datasets...")
    try:
        data_files = {
            "train": "data/processed/train.jsonl",
            "test": "data/processed/test.jsonl"
        }
        dataset = load_dataset("json", data_files=data_files)
        logger.info(f"Dataset loaded: {dataset}")
        
        if len(dataset["train"]) == 0:
            raise ValueError("Training dataset is empty")
        if len(dataset["test"]) == 0:
            raise ValueError("Test dataset is empty")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {str(e)}")
    
    # 预处理数据集
    logger.info("Preprocessing datasets...")
    try:
        tokenized_datasets = dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            batch_size=4,
            remove_columns=dataset["train"].column_names
        )
        logger.info(f"Preprocessing completed. Train size: {len(tokenized_datasets['train'])}")
        
        if len(tokenized_datasets["train"]) == 0:
            raise ValueError("No valid examples in processed training dataset")
            
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess datasets: {str(e)}")
    
    # 训练参数
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("checkpoints", f"qwen_lora_{current_time}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # 减少训练轮数
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,  # 增加梯度累积步数
        learning_rate=1e-4,  # 降低学习率
        weight_decay=0.01,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",  # 使用32bit优化器
        max_grad_norm=0.3,  # 添加梯度裁剪
        lr_scheduler_type="cosine",  # 使用cosine学习率调度
        report_to=["wandb"],
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False  # 禁用pin_memory以节省显存
    )
    
    # 初始化trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=default_data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    logger.info("Starting training...")
    try:
        trainer.train()
        
        # 保存模型
        logger.info("Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        sys.exit(1) 