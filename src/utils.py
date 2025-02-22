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
        examples: 包含system和conversation字段的样本
        tokenizer: 用于tokenize文本的tokenizer
    Returns:
        处理后的样本，包含input_ids和labels
    """
    max_length = 512  # 最大序列长度
    conversations = []
    
    # 确保examples是字典类型
    if isinstance(examples, dict):
        batch_size = len(examples["system"])
    else:
        print(f"Warning: examples is not a dict, type: {type(examples)}")
        return {"input_ids": [], "labels": []}

    # 处理每个样本
    for i in range(batch_size):
        try:
            system = examples["system"][i]
            conv = examples["conversation"][i]
            
            if not isinstance(conv, list):
                print(f"Warning: conversation is not a list, type: {type(conv)}")
                continue
                
            # 构建完整对话
            full_conversation = f"System: {system}\n"
            for turn in conv:
                if not isinstance(turn, dict):
                    print(f"Warning: turn is not a dict, type: {type(turn)}")
                    continue
                    
                if "human" in turn and "assistant" in turn:
                    full_conversation += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"
            
            conversations.append(full_conversation)
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    if not conversations:
        print("Warning: No valid conversations processed")
        return {"input_ids": [], "labels": []}
        
    # Tokenize conversations
    tokenized = tokenizer(
        conversations,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["input_ids"].clone()
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