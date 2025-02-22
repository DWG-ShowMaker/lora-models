import os
import sys
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
from src.utils import (
    setup_logging,
    set_random_seed,
    load_dataset,
    preprocess_function,
    compute_metrics,
)

def main():
    # 设置日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("checkpoints", f"qwen_lora_{current_time}")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    
    # 设置随机种子
    set_random_seed(config.seed)
    
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
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
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
    logger.info("Loading dataset")
    train_dataset, eval_dataset = load_dataset()
    
    # 数据预处理
    logger.info("Preprocessing dataset")
    try:
        # 检查数据集格式
        logger.info(f"Train dataset columns: {train_dataset.column_names}")
        logger.info(f"Train dataset features: {train_dataset.features}")
        
        # 首先处理一个小批量样本进行验证
        logger.info("Validating preprocessing with a small batch...")
        sample_size = min(5, len(train_dataset))
        sample_data = train_dataset.select(range(sample_size))
        
        # 确保数据集格式正确
        if 'system' not in sample_data.column_names or 'conversation' not in sample_data.column_names:
            raise ValueError(f"Dataset missing required columns. Found columns: {sample_data.column_names}")
        
        sample_dict = {
            "system": sample_data["system"],
            "conversation": sample_data["conversation"]
        }
        
        sample_processed = preprocess_function(
            sample_dict,
            tokenizer,
            config.max_seq_length
        )
        logger.info("Sample preprocessing successful")
        
        # 如果验证成功，处理完整数据集
        logger.info("Processing full dataset...")
        
        # 定义预处理函数包装器
        def preprocess_wrapper(examples):
            try:
                return preprocess_function(examples, tokenizer, config.max_seq_length)
            except Exception as e:
                logger.error(f"Error in preprocess_wrapper: {str(e)}")
                raise
        
        # 处理训练集
        logger.info("Processing training dataset...")
        train_dataset = train_dataset.map(
            preprocess_wrapper,
            batched=True,
            batch_size=16,  # 使用更小的批量大小
            num_proc=1,  # 使用单进程以便调试
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train dataset",
            load_from_cache_file=False  # 禁用缓存以便调试
        )
        
        # 处理评估集
        logger.info("Processing evaluation dataset...")
        eval_dataset = eval_dataset.map(
            preprocess_wrapper,
            batched=True,
            batch_size=16,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
            desc="Preprocessing eval dataset",
            load_from_cache_file=False
        )
        
        # 验证处理后的数据集
        logger.info(f"Processed train dataset size: {len(train_dataset)}")
        logger.info(f"Processed eval dataset size: {len(eval_dataset)}")
        
        if len(train_dataset) == 0:
            raise ValueError("No valid examples in processed training dataset")
        
        # 检查处理后的数据集格式
        logger.info(f"Processed train dataset features: {train_dataset.features}")
        if len(train_dataset) > 0:
            logger.info(f"Sample processed input shape: {train_dataset[0]['input_ids'].shape}")
            logger.info(f"Sample processed labels shape: {train_dataset[0]['labels'].shape}")
        
    except Exception as e:
        logger.error(f"Error during dataset preprocessing: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    logger.info("Starting training")
    trainer.train()
    
    # 保存最终模型
    logger.info("Saving final model")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 