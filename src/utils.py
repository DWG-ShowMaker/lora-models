import logging
import os
import random
import numpy as np
import torch
from transformers import set_seed
from modelscope.msdatasets import MsDataset

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
    """加载ModelScope数据集"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Loading training dataset...")
        train_dataset = MsDataset.load(
            'Moemuu/Muice-Dataset',
            subset_name='default',
            split='train',
            cache_dir='data/processed'
        )
        
        logger.info("Loading evaluation dataset...")
        eval_dataset = MsDataset.load(
            'Moemuu/Muice-Dataset',
            subset_name='default',
            split='test',
            cache_dir='data/processed'
        )
        
        # 验证数据集格式
        if not train_dataset or len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if not eval_dataset or len(eval_dataset) == 0:
            raise ValueError("Evaluation dataset is empty")
            
        # 打印数据集信息
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # 检查数据集格式
        sample = next(iter(train_dataset))
        logger.info(f"Dataset sample keys: {list(sample.keys())}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_function(examples, tokenizer, max_length):
    """预处理数据集"""
    # 这里需要根据实际数据集格式调整
    try:
        if isinstance(examples["text"], str):
            texts = [examples["text"]]
        elif isinstance(examples["text"], list):
            texts = examples["text"]
        else:
            raise ValueError(f"Unexpected text format: {type(examples['text'])}")
        
        model_inputs = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 如果是单个样本，去掉批次维度
        if len(texts) == 1:
            model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}
            
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