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
    train_dataset = MsDataset.load('Moemuu/Muice-Dataset', 
                                 subset_name='default', 
                                 split='train')
    eval_dataset = MsDataset.load('Moemuu/Muice-Dataset', 
                                subset_name='default', 
                                split='test')
    return train_dataset, eval_dataset

def preprocess_function(examples, tokenizer, max_length):
    """预处理数据集"""
    # 这里需要根据实际数据集格式调整
    model_inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return model_inputs

def compute_metrics(eval_preds):
    """计算评估指标"""
    # 这里需要根据实际任务调整评估指标
    predictions, labels = eval_preds
    predictions = predictions.argmax(axis=-1)
    
    # 示例：计算准确率
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": accuracy,
    }

def save_model_checkpoint(model, tokenizer, output_dir, step=None):
    """保存模型检查点"""
    if step is not None:
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
    
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir 