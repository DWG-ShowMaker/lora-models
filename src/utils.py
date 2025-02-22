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

def preprocess_function(examples, tokenizer, max_length):
    """预处理数据集"""
    try:
        # 处理对话数据
        conversations = []
        for system, conv in zip(examples["system"], examples["conversation"]):
            # 将system prompt和对话组合在一起
            full_conversation = f"<|im_start|>system\n{system}\n<|im_end|>\n"
            
            # 处理对话列表
            for turn in conv:
                if "human" in turn:
                    full_conversation += f"<|im_start|>user\n{turn['human']}\n<|im_end|>\n"
                if "assistant" in turn:
                    full_conversation += f"<|im_start|>assistant\n{turn['assistant']}\n<|im_end|>\n"
            
            conversations.append(full_conversation)
        
        # 使用tokenizer处理文本
        model_inputs = tokenizer(
            conversations,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 设置labels与input_ids相同（用于自回归训练）
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

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