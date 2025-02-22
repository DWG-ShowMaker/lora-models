import os
import sys
import logging
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
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
        
        # 使用8bit量化加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 启用梯度检查点
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {str(e)}")
    
    # 配置LoRA
    logger.info("Configuring LoRA")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
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
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=1,  # 减小批处理大小
        gradient_accumulation_steps=16,  # 增加梯度累积步数
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        gradient_checkpointing=True,  # 启用梯度检查点
        optim="paged_adamw_8bit"  # 使用8bit优化器
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