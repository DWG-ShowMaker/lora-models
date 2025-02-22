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
    logger = logging.getLogger(__name__)
    try:
        # 处理对话数据
        conversations = []
        
        # 获取批次大小
        if isinstance(examples, dict):
            batch_size = len(examples['system'])
            logger.info(f"Processing batch of size: {batch_size}")
            
            # 遍历批次中的每个样本
            for idx in range(batch_size):
                try:
                    # 获取system和conversation
                    system = examples['system'][idx]
                    conv = examples['conversation'][idx]
                    
                    # 将system prompt和对话组合在一起
                    full_conversation = f"<|im_start|>system\n{system}\n<|im_end|>\n"
                    
                    # 处理对话列表
                    if isinstance(conv, (str, bytes)):
                        # 如果conv是字符串，尝试解析JSON
                        try:
                            conv = json.loads(conv)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse conversation JSON at index {idx}")
                            continue
                    
                    # 确保conv是列表
                    if not isinstance(conv, list):
                        conv = [conv]
                    
                    for turn in conv:
                        if isinstance(turn, dict):
                            if "human" in turn:
                                full_conversation += f"<|im_start|>user\n{turn['human']}\n<|im_end|>\n"
                            if "assistant" in turn:
                                full_conversation += f"<|im_start|>assistant\n{turn['assistant']}\n<|im_end|>\n"
                    
                    conversations.append(full_conversation)
                    
                    # 每处理100个样本记录一次进度
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1} conversations")
                    
                except Exception as e:
                    logger.error(f"Error processing conversation at index {idx}: {str(e)}")
                    continue
        
        if not conversations:
            logger.warning("No valid conversations processed in this batch")
            # 返回空的张量，保持与预期输出格式一致
            return {
                "input_ids": torch.zeros((0, max_length), dtype=torch.long),
                "attention_mask": torch.zeros((0, max_length), dtype=torch.long),
                "labels": torch.zeros((0, max_length), dtype=torch.long)
            }
        
        logger.info(f"Successfully processed {len(conversations)} conversations")
        
        # 使用tokenizer处理文本
        logger.info("Tokenizing conversations...")
        model_inputs = tokenizer(
            conversations,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 设置labels与input_ids相同（用于自回归训练）
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        logger.info(f"Tokenization completed. Input shape: {model_inputs['input_ids'].shape}")
        return model_inputs
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        logger.error(f"Error details: {type(e).__name__}")
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