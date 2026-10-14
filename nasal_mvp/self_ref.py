from __future__ import annotations

from typing import Any

from .corpus import BASELINE_CHARS, TARGET_CHARS, expected_from_prompt


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def classify_rows(rows: list[dict[str, Any]], prompt: str = "") -> tuple[list, list]:
    tchars, bchars = expected_from_prompt(prompt)
    tset = frozenset(tchars) if tchars else TARGET_CHARS
    bset = frozenset(bchars) if bchars else BASELINE_CHARS
    target = [r for r in rows if (r.get("keyword") or "")[:1] in tset]
    baseline = [r for r in rows if (r.get("keyword") or "")[:1] in bset]
    return target, baseline


def self_ref_from_rows(rows: list[dict[str, Any]], prompt: str = "") -> dict[str, Any]:
    """Score = E_target / (E_target + E_baseline) * 100. Not Nasometer nasalance."""
    target, baseline = classify_rows(rows, prompt)
    et = _mean(target, "rms_energy")
    eb = _mean(baseline, "rms_energy")
    score = None
    if et is not None and eb is not None and (et + eb) > 0:
        score = round(100.0 * et / (et + eb), 1)

    f1_t, f1_b = _mean(target, "energy_below_1khz_ratio"), _mean(baseline, "energy_below_1khz_ratio")
    f1_bias = None
    if f1_t is not None and f1_b not in (None, 0):
        f1_bias = round(100.0 * (f1_t - f1_b) / abs(f1_b), 1)

    band = "无鼻音锚点，仅出口腔切片"
    if score is None:
        hint = "本句缺少鼻音基线字（妈/买/面/奶等），无法做自参照。请改读混合句，如「大妈买大马」。"
    elif score < 30:
        band, hint = "参考带：偏低（讨论用 <30）", "算法提示偏低鼻化能量，须听感确认，非诊断。"
    elif score <= 40:
        band, hint = "参考带：中间（讨论用 30–40）", "算法提示中等，须听感确认，非诊断。"
    else:
        band, hint = "参考带：偏高（讨论用 >40）", "算法提示口腔目标窗能量相对鼻音基线偏高，须听感确认，非诊断。"

    return {
        "self_ref_score_0_100": score,
        "formula": "E(口腔目标窗) / [E(口腔目标窗)+E(鼻音基线窗)] × 100",
        "not_nasometer_nasalance": True,
        "n_target_windows": len(target),
        "n_baseline_windows": len(baseline),
        "energy_target": None if et is None else round(et, 8),
        "energy_baseline": None if eb is None else round(eb, 8),
        "f1_energy_bias_pct": f1_bias,
        "f1_bandwidth_hz_target": None if _mean(target, "f1_bandwidth_hz") is None else round(_mean(target, "f1_bandwidth_hz"), 1),
        "hf_db_per_octave_target": _mean(target, "hf_db_per_octave_2k_4k"),
        "band_note": band,
        "algorithmic_hint": hint,
        "self_ref_index": score,  # 兼容旧字段
    }
