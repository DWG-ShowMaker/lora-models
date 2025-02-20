import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
import logging
import json
from modelscope.msdatasets import MsDataset

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

def prepare_directories():
    """创建必要的目录"""
    dirs = ['data/raw', 'data/processed', 'checkpoints', 'logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logging.info(f"Created directory: {d}")

def download_model(model_name="qwen/Qwen2.5-7B", cache_dir="checkpoints"):
    """下载模型和tokenizer"""
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading model {model_name}")
    
    try:
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        logger.info("Successfully downloaded tokenizer")
        
        # 下载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        logger.info("Successfully downloaded model")
        
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def prepare_dataset(output_dir="data/processed"):
    """准备训练数据集"""
    # 数据集使用Muice-Dataset
    train_ds =  MsDataset.load('Moemuu/Muice-Dataset', subset_name='default', split='train')
    test_ds =  MsDataset.load('Moemuu/Muice-Dataset', subset_name='default', split='test')
    # 保存为jsonl格式
    train_ds.save_to_file(os.path.join(output_dir, "train.jsonl"))
    test_ds.save_to_file(os.path.join(output_dir, "test.jsonl"))
    return train_ds, test_ds

def main():
    # 设置日志
    logger = setup_logging()
    
    try:
        # 创建目录
        prepare_directories()
        
        # 下载模型
        tokenizer, model = download_model()
        
        # 准备数据集
        dataset = prepare_dataset()
        
        logger.info("All preparation steps completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 