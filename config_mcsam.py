# config_integrated.py
"""
集成模型的配置文件
"""

import os
from dataclasses import dataclass


@dataclass
class IntegratedModelConfig:
    """集成模型配置"""
    # 模型基础配置
    model_type: str = "vit_h"
    image_size: int = 1024
    patch_size: int = 16
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16

    # 集成模块配置
    use_rankdice: bool = True
    use_hypercond: bool = True
    n_streams: int = 4
    prompt_dim: int = 256
    cond_dim: int = 128
    cspg_temperature: float = 1.0
    cspg_iters: int = 5

    # 超参数条件化范围
    threshold_range: tuple = (0.4, 0.6)
    boundary_weight_range: tuple = (0.3, 1.0)

    # 训练配置
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 0.00005
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # 路径配置
    work_dir: str = "./work_dir"
    model_save_path: str = "./checkpoints"
    log_dir: str = "./logs"

    def __post_init__(self):
        """初始化后处理"""
        dimensions = {"vit_b": (768, 12, 12), "vit_l": (1024, 24, 16), "vit_h": (1280, 32, 16)}
        self.embed_dim, self.depth, self.num_heads = dimensions[self.model_type]
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        return cls(
            model_type=args.model_type,
            cspg_temperature=getattr(args, "cspg_temperature", 1.0),
            cspg_iters=getattr(args, "cspg_iters", 5),
            use_rankdice=args.use_rankdice,
            use_hypercond=args.use_hypercond,
            n_streams=args.n_streams,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            work_dir=args.work_dir,
        )


# 默认配置
default_config = IntegratedModelConfig()