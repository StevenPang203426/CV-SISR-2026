"""
models 包 —— 统一的模型注册表与工厂函数
=========================================

所有模型在此集中注册，外部通过 ``build_model()`` 创建实例，
不再需要在每个入口脚本里重复导入和实例化逻辑。

用法::

    from models import build_model, REGISTRY

    model = build_model('edsr', scale=4, in_channels=3)
    print(list(REGISTRY.keys()))   # ['srcnn', 'fsrcnn', 'espcn', 'edsr', 'imdn']
"""

import inspect
from .srcnn import SRCNN
from .fsrcnn import FSRCNN
from .espcn import ESPCN
from .edsr import EDSR
from .imdn import IMDN

# ---- 注册表 ----
REGISTRY: dict[str, type] = {
    'srcnn':  SRCNN,
    'fsrcnn': FSRCNN,
    'espcn':  ESPCN,
    'edsr':   EDSR,
    'imdn':   IMDN,
}


def build_model(name: str, **kwargs):
    """
    根据名称和参数构建模型实例。

    该函数会自动过滤掉目标模型不接受的参数，因此可以安全地传入
    通用参数（如 ``scale``、``in_channels``），无需担心某个模型
    不支持某个参数而报错。

    Parameters
    ----------
    name : str
        模型名称，必须是 REGISTRY 中的键之一。
    **kwargs
        传递给模型构造函数的参数。不被目标模型接受的参数会被静默忽略。

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    ValueError
        当 ``name`` 不在 REGISTRY 中时抛出。

    Examples
    --------
    >>> model = build_model('srcnn', scale=2, in_channels=3)
    >>> model = build_model('edsr', scale=4, n_feats=64, n_resblocks=16)
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(REGISTRY.keys())}"
        )
    cls = REGISTRY[name]

    # 自省：只保留目标构造函数实际接受的参数
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}

    # 如果构造函数有 **kwargs，则不做过滤
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**filtered)
