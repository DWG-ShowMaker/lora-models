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
    
    # 使用ModelScope下载模型
    logger.info(f"Loading model from {config.model_name_or_path}")
    model_dir = snapshot_download('qwen/Qwen2.5-7B', cache_dir='checkpoints')
    
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
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        num_proc=config.preprocessing_num_workers,
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_seq_length),
        batched=True,
        num_proc=config.preprocessing_num_workers,
    )
    
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