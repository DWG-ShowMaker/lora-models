from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # 模型配置
    model_name_or_path: str = "qwen/Qwen1.5-7B"
    use_auth_token: bool = True

    # LoRA配置
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 训练配置
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # 数据配置
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    
    # 其他配置
    seed: int = 42
    fp16: bool = True
    logging_steps: int = 10
    report_to: list = None

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if self.report_to is None:
            self.report_to = ["tensorboard", "wandb"]

config = TrainingConfig() 