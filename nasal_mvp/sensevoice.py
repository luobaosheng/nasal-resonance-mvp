"""SenseVoiceSmall local ASR + timestamps. Optional extra; not required for demo/analyze."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".modelscope_cache"

# 词表草案里的高压口辅音 / 对照字；可 --keywords 覆盖
DEFAULT_KEYWORDS = tuple("爸怕打靶哥骑车大吃阿姨鱼乌鸦喝水皮球开花朵看书妈买马面奶")

_MODEL = None


def _device() -> str:
    return os.environ.get("NASAL_MVP_DEVICE", "cpu")


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    os.environ.setdefault("MODELSCOPE_CACHE", str(CACHE))
    os.environ.setdefault("MODELSCOPE_HUB", os.environ.get("MODELSCOPE_HUB", "modelscope"))
    from funasr import AutoModel

    print(f"Loading SenseVoiceSmall on {_device()} (first run downloads weights)...")
    _MODEL = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=_device(),
        disable_update=True,
        trust_remote_code=True,
    )
    return _MODEL


def _strip_tags(text: str) -> str:
    return re.sub(r"<\|[^>]+\|>", "", text or "").replace(" ", "").strip()


def _as_ms(a: float, b: float) -> tuple[float, float]:
    # SenseVoiceSmall on this FunASR build returns milliseconds already
    # (e.g. 210–270). Only treat as seconds if both ends look like seconds.
    if 0 <= a < 20 and 0 < b < 20:
        return a * 1000.0, b * 1000.0
    return a, b


def _pairs_from_result(res0: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize FunASR output into [{char, start_ms, end_ms}, ...]."""
    pairs: list[dict[str, Any]] = []
    ts = res0.get("timestamp")
    words = res0.get("words")
    text = _strip_tags(str(res0.get("text") or res0.get("sentence") or ""))

    if words and ts and len(words) == len(ts):
        for tok, item in zip(words, ts):
            a, b = float(item[0]), float(item[1])
            a, b = _as_ms(a, b)
            ch = _strip_tags(str(tok))
            if ch:
                pairs.append({"char": ch, "start_ms": a, "end_ms": b})
        return pairs

    if ts and isinstance(ts, list) and ts:
        first = ts[0]
        if isinstance(first, (list, tuple)) and len(first) >= 3 and isinstance(first[0], str):
            for item in ts:
                tok, a, b = item[0], float(item[1]), float(item[2])
                a, b = _as_ms(a, b)
                pairs.append({"char": _strip_tags(tok) or tok, "start_ms": a, "end_ms": b})
            return [p for p in pairs if p["char"]]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            chars = list(text.replace(" ", ""))
            for i, item in enumerate(ts):
                a, b = _as_ms(float(item[0]), float(item[1]))
                ch = chars[i] if i < len(chars) else ""
                if ch:
                    pairs.append({"char": ch, "start_ms": a, "end_ms": b})
            return pairs

    for seg in res0.get("sentence_info") or []:
        st = float(seg.get("start") or 0)
        en = float(seg.get("end") or 0)
        sent = _strip_tags(str(seg.get("text") or seg.get("sentence") or ""))
        if sent:
            pairs.append({"char": sent, "start_ms": st, "end_ms": en, "is_segment": True})
        nested = seg.get("timestamp")
        if nested:
            fake = {"text": sent, "timestamp": nested}
            pairs.extend(_pairs_from_result(fake))
    if pairs:
        return pairs
    if text:
        pairs.append({"char": text, "start_ms": 0, "end_ms": None, "is_segment": True})
    return pairs


def transcribe(wav_path: Path) -> dict[str, Any]:
    model = load_model()
    wav_path = Path(wav_path)
    raw = model.generate(
        input=str(wav_path),
        cache={},
        language="zh",
        use_itn=False,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        output_timestamp=True,
    )
    res0 = raw[0] if raw else {}
    if isinstance(res0, list):
        res0 = res0[0] if res0 else {}
    pairs = _pairs_from_result(res0)
    return {
        "text": _strip_tags(str(res0.get("text") or "")),
        "units": pairs,
        "raw_keys": sorted(res0.keys()) if isinstance(res0, dict) else [],
    }


def find_keywords(
    units: list[dict[str, Any]],
    keywords: tuple[str, ...] = DEFAULT_KEYWORDS,
) -> list[dict[str, Any]]:
    hits = []
    for u in units:
        ch = u.get("char") or ""
        for kw in keywords:
            if kw and kw in ch:
                hits.append({**u, "keyword": kw})
                break
    return hits
