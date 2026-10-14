from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
CASES = OUT / "cases"
INDEX = CASES / "index.json"

FEATURE_ZH = [
    ("duration_s", "时长 (秒)"),
    ("f0_median_hz", "基频 F0 中位 (Hz)"),
    ("f1_median_hz", "第一共振峰 F1 (Hz)"),
    ("f2_median_hz", "第二共振峰 F2 (Hz)"),
    ("f3_median_hz", "第三共振峰 F3 (Hz)"),
    ("hnr_mean_db", "谐波噪声比 HNR (dB)"),
    ("spectral_centroid_hz", "频谱重心 (Hz)"),
    ("energy_below_1khz_ratio", "1 kHz 以下能量比"),
    ("hf_slope_2_4khz", "2–4 kHz 能量斜率 (/kHz)"),
]


def safe_code(code: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", (code or "").strip())
    return s[:40] or "未编号"


def format_features_zh(rec: dict, extra: dict | None = None) -> str:
    feat = rec.get("features") or {}
    perc = rec.get("perceptual") or {}
    lines = ["【声学特征（人工可读，非程序源码）】", ""]
    for key, label in FEATURE_ZH:
        val = feat.get(key)
        lines.append(f"{label}：{val if val is not None else '—'}")
    lines += [
        "",
        f"听感评分人：{perc.get('rater') or '—'}",
        f"过高鼻音 0–3：{perc.get('hypernasality_0_3') if perc.get('hypernasality_0_3') is not None else '未填'}",
        "说明：以上不是鼻音率（nasalance），不能替代诊断。",
    ]
    return "\n".join(lines)


def format_align_zh(extra: dict | None, csv_tail: str = "", note: str = "") -> str:
    extra = extra or {}
    score = extra.get("self_ref_score_0_100")
    if score is None:
        score = extra.get("self_ref_index")
    lines = [
        "VPI 鼻音客观量化评估（研究用，非诊断）",
        "",
        "一、评估基础信息",
        f"测试文本：{extra.get('prompt') or '—'}",
        f"语料编号：{extra.get('sentence_id') or '—'}",
        f"切片成功率：{extra.get('slice_success') or '—'}（SenseVoice 字级窗，非音素金标准）",
        f"信噪比预检 SNR：{extra.get('snr_db') if extra.get('snr_db') is not None else '—'} dB",
        "",
        "二、核心声学量化指标",
        f"自参照指数 (0–100)：{score if score is not None else '—'}",
        "公式：E(口腔目标窗) / [E(口腔目标)+E(鼻音基线)] × 100",
        "说明：同一段录音内相除，减轻麦克风增益；不是 Nasometer 鼻音率。",
        "参考带（仅讨论，非器械阈值）：偏低 <30，中间 30–40，偏高 >40。",
        f"分带说明：{extra.get('band_note') or '—'}",
        f"口腔目标窗条数：{extra.get('n_target_windows', extra.get('n_oral_windows', '—'))}",
        f"鼻音基线窗条数：{extra.get('n_baseline_windows', extra.get('n_nasal_char_windows', '—'))}",
        f"F1 能量偏置比：{extra.get('f1_energy_bias_pct') if extra.get('f1_energy_bias_pct') is not None else '—'} %",
        f"目标窗 F1 带宽：{extra.get('f1_bandwidth_hz_target') if extra.get('f1_bandwidth_hz_target') is not None else '—'} Hz",
        f"目标窗 2–4 kHz 衰减：{extra.get('hf_db_per_octave_target') if extra.get('hf_db_per_octave_target') is not None else '—'} dB/倍频",
        "",
        "三、算法提示（须听感确认，不替代内镜/鼻音计）",
        extra.get("algorithmic_hint") or "—",
    ]
    if note:
        lines += ["", note]
    if (csv_tail or "").strip():
        lines += ["", "四、切窗明细", csv_tail.strip()]
    return "\n".join(lines)


def _load_index() -> list[dict]:
    if not INDEX.exists():
        return []
    return json.loads(INDEX.read_text(encoding="utf-8"))


def _save_index(rows: list[dict]) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_subject(code: str, name: str, rater: str) -> dict:
    code = safe_code(code)
    rows = _load_index()
    now = datetime.now().isoformat(timespec="seconds")
    found = next((r for r in rows if r["code"] == code), None)
    if found:
        found["name"] = name or found.get("name") or ""
        found["rater"] = rater or found.get("rater") or ""
        found["updated"] = now
    else:
        found = {"code": code, "name": name or "", "rater": rater or "", "created": now, "updated": now}
        rows.append(found)
    _save_index(rows)
    (CASES / code / "audio").mkdir(parents=True, exist_ok=True)
    (CASES / code / "analysis").mkdir(parents=True, exist_ok=True)
    meta = CASES / code / "meta.json"
    meta.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    return found


def subject_choices() -> list[str]:
    rows = sorted(_load_index(), key=lambda r: r.get("updated") or "", reverse=True)
    out = []
    for r in rows:
        label = f"{r['code']}  {r.get('name') or '（未填姓名）'}"
        out.append(label)
    return out


def code_from_choice(choice: str | None) -> str:
    if not choice:
        return ""
    return safe_code(choice.split()[0])


def save_audio(code: str, src: Path, label: str) -> Path:
    code = safe_code(code)
    folder = CASES / code / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = safe_code(label) or "未标任务"
    dest = folder / f"{stamp}_{tag}{src.suffix or '.wav'}"
    shutil.copy2(src, dest)
    return dest


def list_audio(code: str) -> list[list[str]]:
    code = safe_code(code)
    folder = CASES / code / "audio"
    if not folder.exists():
        return []
    rows = []
    for p in sorted(folder.glob("*"), reverse=True):
        if p.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            continue
        rows.append([p.name, p.stem.split("_", 2)[-1] if "_" in p.stem else "", str(p)])
    return rows


def audio_path(code: str, filename: str) -> Path | None:
    p = CASES / safe_code(code) / "audio" / filename
    return p if p.exists() else None
