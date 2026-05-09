"""
配置加载模块
=============

从 YAML 文件加载训练/测试配置，将嵌套字典扁平化为 argparse.Namespace。
"""

import os
import argparse
import yaml


def load_config(extra_args=None):
    """
    解析命令行参数并从 YAML 配置文件加载参数。

    YAML 中的嵌套字典（如 ``train:`` 下的子键）会被扁平化到
    Namespace 的顶层属性中。

    Parameters
    ----------
    extra_args : list[str], optional
        额外的命令行参数（主要用于测试）。

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args, remaining = parser.parse_known_args(extra_args)

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    setattr(args, kk, vv)
            else:
                setattr(args, k, v)

    return args
