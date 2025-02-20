import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.lora_config import config
from src.utils import setup_logging, load_dataset

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        device_map="auto"
    )
    
    # 加载LoRA权重
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """生成回复"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_model(model_path, output_dir):
    """评估模型"""
    logger = setup_logging(output_dir)
    logger.info(f"Loading model from {model_path}")
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 加载测试数据集
    _, eval_dataset = load_dataset()
    
    # 评估结果
    results = []
    
    # 对每个测试样本进行评估
    logger.info("Starting evaluation")
    for i, example in enumerate(tqdm(eval_dataset)):
        prompt = example["text"]  # 根据实际数据集格式调整
        response = generate_response(model, tokenizer, prompt)
        
        result = {
            "id": i,
            "prompt": prompt,
            "response": response,
        }
        results.append(result)
    
    # 保存评估结果
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation completed. Results saved to {output_file}")

def main():
    # 获取最新的模型检查点
    checkpoints_dir = "checkpoints"
    model_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith("qwen_lora_")]
    if not model_dirs:
        raise ValueError("No trained model checkpoints found!")
    
    latest_model = sorted(model_dirs)[-1]
    model_path = os.path.join(checkpoints_dir, latest_model)
    
    # 创建评估输出目录
    eval_output_dir = os.path.join(model_path, "evaluation")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # 开始评估
    evaluate_model(model_path, eval_output_dir)

if __name__ == "__main__":
    main() 