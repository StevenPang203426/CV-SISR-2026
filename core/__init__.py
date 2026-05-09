"""
core 包 —— 训练、评估、推理的核心引擎
========================================

将原先散布在 train.py / test.py / infer.py 中的业务逻辑
抽取为可复用的组件，入口脚本只负责参数解析和调用。
"""
from .config import load_config
from .checkpoint import save_checkpoint, load_checkpoint
from .trainer import Trainer
from .evaluator import Evaluator
from .inferencer import Inferencer

__all__ = [
    'load_config',
    'save_checkpoint', 'load_checkpoint',
    'Trainer', 'Evaluator', 'Inferencer',
]
