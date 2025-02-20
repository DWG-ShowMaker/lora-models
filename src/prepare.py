import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer
import logging
import json

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
    logger = logging.getLogger(__name__)
    logger.info("Preparing dataset")
    
    # 示例对话数据
    conversations = [
        {
            "conversations": [
                {"role": "user", "content": "你好，请介绍一下你自己"},
                {"role": "assistant", "content": "你好！我是一个AI助手，我可以帮助你回答问题、解决问题，并与你进行友好的对话。"}
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "什么是机器学习？"},
                {"role": "assistant", "content": "机器学习是人工智能的一个分支，它使计算机系统能够通过经验自动改进。简单来说，它让计算机能够从数据中学习，而不需要被明确编程。"}
            ]
        }
    ]
    
    # 保存为jsonl格式
    output_file = os.path.join(output_dir, "train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    logger.info(f"Created example dataset at {output_file}")
    
    # 加载数据集
    dataset = load_dataset("json", data_files=output_file)
    logger.info(f"Dataset loaded with {len(dataset['train'])} examples")
    
    return dataset

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