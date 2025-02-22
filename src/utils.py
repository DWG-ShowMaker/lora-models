import logging
import os
import random
import numpy as np
import torch
from transformers import set_seed
from modelscope.msdatasets import MsDataset
import json
from datasets import load_dataset as hf_load_dataset

def setup_logging(save_path):
    """设置日志配置"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(save_path, "train.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def load_dataset():
    """加载数据集"""
    logger = logging.getLogger(__name__)
    
    try:
        # 检查本地数据集文件
        train_file = os.path.join("data/processed", "train.jsonl")
        test_file = os.path.join("data/processed", "test.jsonl")
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            logger.info("Loading local datasets...")
            train_dataset = hf_load_dataset('json', data_files={'train': train_file})['train']
            test_dataset = hf_load_dataset('json', data_files={'train': test_file})['train']
            
            logger.info(f"Loaded local datasets - Train: {len(train_dataset)} examples, Test: {len(test_dataset)} examples")
            return train_dataset, test_dataset
        
        # 如果本地文件不存在，从ModelScope下载
        logger.info("Local datasets not found, downloading from ModelScope...")
        train_dataset = MsDataset.load(
            'Moemuu/Muice-Dataset',
            subset_name='default',
            split='train',
            cache_dir='data/processed'
        )
        test_dataset = MsDataset.load(
            'Moemuu/Muice-Dataset',
            subset_name='default',
            split='test',
            cache_dir='data/processed'
        )
        
        # 保存为jsonl格式
        logger.info("Saving datasets to local files...")
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Datasets saved to {train_file} and {test_file}")
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_function(examples, tokenizer):
    """
    预处理函数，用于处理数据集中的样本。
    Args:
        examples: LazyBatch 对象，包含 system 和 conversation 字段
        tokenizer: 用于 tokenize 文本的 tokenizer
    Returns:
        处理后的样本，包含 input_ids 和 labels
    """
    # 获取logger
    logger = logging.getLogger(__name__)
    
    max_length = 512  # 最大序列长度
    conversations = []
    
    # 获取批次大小
    batch_size = len(examples["system"])
    logger.info(f"Processing batch of size: {batch_size}")
    
    # 处理每个样本
    for i in range(batch_size):
        try:
            # 获取system和conversation
            system = examples["system"][i]
            conversation = examples["conversation"][i]
            
            # 构建完整对话
            full_conversation = f"<|im_start|>system\n{system}\n<|im_end|>\n"
            
            # 处理对话
            for turn in conversation:
                if isinstance(turn, dict) and "human" in turn and "assistant" in turn:
                    full_conversation += f"<|im_start|>user\n{turn['human']}\n<|im_end|>\n"
                    full_conversation += f"<|im_start|>assistant\n{turn['assistant']}\n<|im_end|>\n"
            
            conversations.append(full_conversation)
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {str(e)}")
            continue
    
    if not conversations:
        logger.warning("No valid conversations processed")
        # 返回空张量，保持与期望的形状一致
        empty_tensor = torch.zeros((0, max_length), dtype=torch.long)
        return {
            "input_ids": empty_tensor,
            "attention_mask": empty_tensor,
            "labels": empty_tensor
        }
    
    # Tokenize conversations
    try:
        # 使用tokenizer处理文本
        tokenized = tokenizer(
            conversations,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 创建标签，复制input_ids
        labels = tokenized["input_ids"].clone()
        
        # 返回处理后的数据
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
        
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        # 返回空张量，保持与期望的形状一致
        empty_tensor = torch.zeros((0, max_length), dtype=torch.long)
        return {
            "input_ids": empty_tensor,
            "attention_mask": empty_tensor,
            "labels": empty_tensor
        }

def compute_metrics(eval_preds):
    """计算评估指标"""
    try:
        predictions, labels = eval_preds
        predictions = predictions.argmax(axis=-1)
        
        # 计算准确率
        accuracy = (predictions == labels).mean()
        
        return {
            "accuracy": float(accuracy),  # 确保是Python原生类型
        }
    except Exception as e:
        logging.error(f"Error computing metrics: {str(e)}")
        raise

def save_model_checkpoint(model, tokenizer, output_dir, step=None):
    """保存模型检查点"""
    try:
        if step is not None:
            output_dir = os.path.join(output_dir, f"checkpoint-{step}")
        
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logging.info(f"Model checkpoint saved to {output_dir}")
        return output_dir
        
    except Exception as e:
        logging.error(f"Error saving checkpoint: {str(e)}")
        raise 