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

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查本地模型
    model_dir = os.path.join("checkpoints", "qwen/Qwen2.5-7B")
    if not os.path.exists(model_dir):
        logger.info(f"Local model not found, downloading from ModelScope...")
        model_dir = snapshot_download('qwen/Qwen2.5-7B', cache_dir='checkpoints')
    else:
        logger.info(f"Using local model from {model_dir}")
    
    # 加载tokenizer和模型
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置LoRA
    logger.info("Configuring LoRA")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    
    # 加载数据集
    logger.info("Loading datasets...")
    data_files = {
        "train": "data/processed/train.jsonl",
        "test": "data/processed/test.jsonl"
    }
    dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Dataset loaded: {dataset}")
    
    # 预处理数据集
    logger.info("Preprocessing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=4,
        remove_columns=dataset["train"].column_names
    )
    logger.info(f"Preprocessing completed. Train size: {len(tokenized_datasets['train'])}")
    
    # 训练参数
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("checkpoints", f"qwen_lora_{current_time}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=config.fp16
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
    trainer.train()
    
    # 保存模型
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 