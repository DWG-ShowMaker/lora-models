from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DeployConfig:
    # 服务配置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # VLLM配置
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 256
    trust_remote_code: bool = True
    
    # 推理配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1
    
    # 模型配置
    base_model_path: str = "qwen/Qwen2.5-7B"
    lora_model_path: Optional[str] = None
    quantization: Optional[str] = None  # 可选值: "4bit", "8bit", None
    
    # 系统配置
    seed: int = 42
    disable_log_stats: bool = False
    disable_log_requests: bool = False

config = DeployConfig() 