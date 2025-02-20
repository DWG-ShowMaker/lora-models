import os
import sys
import torch
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import logging

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.deploy_config import config
from src.utils import setup_logging

# 创建FastAPI应用
app = FastAPI(
    title="Qwen LoRA API",
    description="使用VLLM部署的Qwen LoRA模型API服务",
    version="0.1.0"
)

# 请求模型
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_return_sequences: Optional[int] = None

# 响应模型
class GenerateResponse(BaseModel):
    responses: List[str]
    usage: dict

# 全局变量
llm = None
logger = None

@app.on_event("startup")
async def startup_event():
    global llm, logger
    
    # 设置日志
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir)
    
    # 加载模型
    logger.info("Loading model...")
    try:
        llm = LLM(
            model=config.base_model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            trust_remote_code=config.trust_remote_code,
            quantization=config.quantization,
            # 如果有LoRA权重，添加加载逻辑
            adapter_path=config.lora_model_path if config.lora_model_path else None,
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    global llm, logger
    
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # 准备采样参数
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens or config.max_new_tokens,
            temperature=request.temperature or config.temperature,
            top_p=request.top_p or config.top_p,
            top_k=request.top_k or config.top_k,
            n=request.num_return_sequences or config.num_return_sequences,
        )
        
        # 生成回复
        outputs = llm.generate(request.prompt, sampling_params)
        
        # 提取生成的文本
        responses = [output.outputs[0].text for output in outputs]
        
        # 计算token使用情况
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        completion_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        total_tokens = prompt_tokens + completion_tokens
        
        return GenerateResponse(
            responses=responses,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": llm is not None}

def main():
    # 启动服务
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main() 