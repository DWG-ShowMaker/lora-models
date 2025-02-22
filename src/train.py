import os
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed
)
from utils import preprocess_function, compute_metrics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 加载tokenizer和模型
    model_name = "THUDM/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    
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
    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True
    )
    
    # 初始化trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        compute_metrics=compute_metrics
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存模型
    logger.info("Saving final model...")
    trainer.save_model("final_model")
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 