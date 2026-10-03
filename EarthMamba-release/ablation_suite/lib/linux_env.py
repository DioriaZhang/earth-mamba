"""Linux 运行环境清理（Windows 同步后去除 CRLF 污染）。"""
from __future__ import annotations

import os
from typing import Any


def clean_str(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("\r", "").strip()


def clean_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return clean_str(obj)
    if isinstance(obj, dict):
        return {clean_str(str(k)): clean_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_obj(v) for v in obj]
    return obj


def normalize_linux_env() -> None:
    """必须在 import torch / 首次 CUDA 调用之前执行。"""
    for key in list(os.environ):
        val = os.environ[key]
        if isinstance(val, str) and "\r" in val:
            os.environ[key] = clean_str(val)


# 任何 from lib.linux_env import ... 时自动清理当前进程环境
normalize_linux_env()
