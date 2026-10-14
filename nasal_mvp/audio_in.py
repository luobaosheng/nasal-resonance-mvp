from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_audio(src: Any) -> Path | None:
    """Gradio Audio 可能是路径字符串、dict 或 (sr, ndarray)。"""
    if src is None:
        return None
    if isinstance(src, Path):
        return src if src.exists() else None
    if isinstance(src, str):
        p = Path(src)
        return p if p.exists() else None
    if isinstance(src, dict):
        for key in ("path", "name", "orig_name"):
            v = src.get(key)
            if v:
                p = Path(str(v))
                if p.exists():
                    return p
        return None
    if isinstance(src, (list, tuple)) and src:
        return resolve_audio(src[0])
    return None
