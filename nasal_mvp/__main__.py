from __future__ import annotations

import argparse
import csv
from pathlib import Path

from . import DISCLAIMER, SCHEMA, __version__
from .features import attach_perceptual, extract_features, write_json
from .synth import write_demo_stereo, write_demo_wavs
from .windows import slice_wav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def _banner() -> None:
    print(f"nasal_mvp {__version__}  schema={SCHEMA}")
    print(DISCLAIMER)


def cmd_demo() -> int:
    wavs = write_demo_wavs(OUT / "synth")
    rows = []
    for w in wavs:
        rec = extract_features(w)
        rec = attach_perceptual(
            rec,
            rater="pipeline_demo",
            hypernasality_0_3=None,
            notes="synthetic vowel; not a patient sample",
        )
        json_path = OUT / "json" / (w.stem + ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(rec, json_path)
        feat = rec["features"]
        rows.append(
            {
                "source_file": rec["source_file"],
                "duration_s": rec["duration_s"],
                **feat,
                "rater": rec["perceptual"]["rater"],
                "hypernasality_0_3": rec["perceptual"]["hypernasality_0_3"],
            }
        )
        print(f"  {w.name}: F1={feat['f1_median_hz']} F2={feat['f2_median_hz']} HNR={feat['hnr_mean_db']}")
    csv_path = OUT / "features.csv"
    _write_csv(csv_path, rows)
    print(f"CSV -> {csv_path}")
    return 0


def cmd_demo_stereo() -> int:
    wavs = write_demo_stereo(OUT / "synth")
    rows = []
    for w in wavs:
        rec = extract_features(w)
        rec = attach_perceptual(
            rec,
            rater="pipeline_demo",
            notes="synthetic stereo L=oral R=nasal; not a patient sample",
        )
        json_path = OUT / "json" / (w.stem + ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(rec, json_path)
        feat = rec["features"]
        rows.append(
            {
                "source_file": rec["source_file"],
                "duration_s": rec["duration_s"],
                **feat,
                "rater": rec["perceptual"]["rater"],
                "hypernasality_0_3": rec["perceptual"]["hypernasality_0_3"],
            }
        )
        print(
            f"  {w.name}: split_energy_index={feat['split_energy_index']} "
            f"(not Nasometer) oral_rms={feat['oral_or_larynx_rms']} nasal_rms={feat['nasal_rms']}"
        )
    csv_path = OUT / "features_stereo.csv"
    _write_csv(csv_path, rows)
    print(f"CSV -> {csv_path}")
    return 0


def cmd_analyze(path: Path, rater: str | None, grade: int | None, notes: str | None, plot: bool) -> int:
    rec = extract_features(path)
    rec = attach_perceptual(rec, rater=rater, hypernasality_0_3=grade, notes=notes)
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "json" / (path.stem + ".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(rec, json_path)
    csv_path = OUT / "features.csv"
    row = {
        "source_file": rec["source_file"],
        "duration_s": rec["duration_s"],
        **rec["features"],
        "rater": rec["perceptual"].get("rater"),
        "hypernasality_0_3": rec["perceptual"].get("hypernasality_0_3"),
    }
    _append_csv(csv_path, row)
    if plot:
        from .plots import spectrogram_2d, spectrogram_3d

        p2 = spectrogram_2d(path, OUT / "plots" / f"{path.stem}_spec2d.png", f"2D 语谱图 {path.name}")
        p3 = spectrogram_3d(path, OUT / "plots" / f"{path.stem}_spec3d.png", f"3D 瀑布图 {path.name}")
        print(p2)
        print(p3)
    print(json_path)
    print(rec["features"])
    return 0


def cmd_align(path: Path, keywords: str | None, prompt: str = "") -> int:
    from .corpus import expected_from_prompt, match_sentence
    from .features import estimate_snr_db
    from .sensevoice import DEFAULT_KEYWORDS, find_keywords, transcribe

    tchars, bchars = expected_from_prompt(prompt)
    if keywords and keywords.strip():
        kws = tuple(x.strip() for x in keywords.split(",") if x.strip())
    else:
        merged = "".join(dict.fromkeys((tchars or "") + (bchars or "")))
        kws = tuple(merged) if merged else DEFAULT_KEYWORDS
    result = transcribe(path)
    print(f"ASR text: {result['text'] or '(empty)'}")
    print(f"raw keys: {result['raw_keys']}")
    units = result["units"]
    if not units:
        print("No timestamp units. SenseVoice returned text only — cannot slice.")
        return 2
    for u in units[:40]:
        print(f"  {u.get('char')!r}  {u.get('start_ms')}–{u.get('end_ms')} ms")
    hits = find_keywords(units, kws)
    print(f"keyword hits ({','.join(kws)}): {len(hits)}")
    win_dir = OUT / "windows" / path.stem
    rows = []
    for i, h in enumerate(hits):
        if h.get("end_ms") is None:
            continue
        dest = win_dir / f"{i:02d}_{h.get('keyword')}.wav"
        kw = str(h.get("keyword") or "")
        role = "target" if kw[:1] in (tchars or "") else ("baseline" if kw[:1] in (bchars or "") else "other")
        try:
            slice_wav(
                path,
                dest,
                float(h["start_ms"]),
                float(h["end_ms"]),
                pad_ms=8 if role == "target" else 20,
                max_dur_ms=80 if role == "target" else 120,
            )
        except ValueError as exc:
            print(f"  skip {h}: {exc}")
            continue
        rec = extract_features(dest)
        rec["alignment"] = {
            "keyword": h.get("keyword"),
            "start_ms": h.get("start_ms"),
            "end_ms": h.get("end_ms"),
            "engine": "SenseVoiceSmall",
            "not_phoneme_gold": True,
        }
        json_path = OUT / "json" / f"{path.stem}_{dest.stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(rec, json_path)
        feat = rec["features"]
        print(
            f"  slice {dest.name}: F1={feat.get('f1_median_hz')} F2={feat.get('f2_median_hz')}"
        )
        rows.append(
            {
                "source_file": rec["source_file"],
                "keyword": h.get("keyword"),
                "role": role,
                "start_ms": h.get("start_ms"),
                "end_ms": h.get("end_ms"),
                **feat,
            }
        )
    if rows:
        csv_path = OUT / f"features_align_{path.stem}.csv"
        _write_csv(csv_path, rows)
        print(f"CSV -> {csv_path}")
        from .self_ref import self_ref_from_rows

        sr = self_ref_from_rows(rows, prompt)
        sent = match_sentence(prompt)
        exp_n = sum(1 for c in (prompt or "") if c in (tchars + bchars))
        hit_n = len(rows)
        sr.update(
            {
                "prompt": prompt or result.get("text") or "",
                "sentence_id": (sent or {}).get("id") if sent else "",
                "asr_text": result.get("text") or "",
                "slice_success": f"{hit_n}/{exp_n}" if exp_n else f"{hit_n}",
                "snr_db": estimate_snr_db(path),
            }
        )
        write_json(sr, OUT / "json" / f"{path.stem}_self_ref.json")
        print(f"self_ref_score={sr.get('self_ref_score_0_100')} (not Nasometer) {sr}")
        try:
            from .plots import envelope_overlay, spectrogram_2d
            from .self_ref import classify_rows

            tgt, base = classify_rows(rows, prompt)
            tfiles = [win_dir / r["source_file"] for r in tgt if (win_dir / r["source_file"]).exists()]
            bfiles = [win_dir / r["source_file"] for r in base if (win_dir / r["source_file"]).exists()]
            if tfiles:
                spectrogram_2d(tfiles[0], OUT / "plots" / f"{path.stem}_target_spec.png", "口腔目标窗语谱（切片）")
            if tfiles and bfiles:
                envelope_overlay(tfiles, bfiles, OUT / "plots" / f"{path.stem}_envelope.png")
        except Exception as exc:
            print("plot skip:", exc)
    return 0


def cmd_align_demo() -> int:
    """macOS `say` Chinese utterance — not a patient sample."""
    import subprocess

    aiff = OUT / "synth" / "say_card.aiff"
    wav = OUT / "synth" / "say_card.wav"
    aiff.parent.mkdir(parents=True, exist_ok=True)
    text = "爸爸怕打靶，妈妈买牛奶"
    subprocess.run(["say", "-v", "Tingting", "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(wav)],
        check=True,
    )
    print(f"TTS wav -> {wav}  text={text!r}")
    return cmd_align(wav, None, text)


def cmd_report() -> int:
    """Phase-1 demo pack: plots + simulated pre/post + PDF. Not patient data."""
    from .plots import spectrogram_3d, spectrogram_pair_2d
    from .report import write_pdf
    from .self_ref import self_ref_from_rows
    from .simulate import simulate_hypernasal

    wav = OUT / "synth" / "say_card.wav"
    if not wav.exists():
        rc = cmd_align_demo()
        if rc != 0:
            return rc
    else:
        cmd_align(wav, None, "爸爸怕打靶，妈妈买牛奶")
    pre = simulate_hypernasal(wav, OUT / "synth" / "say_card_sim_pre.wav")
    plots = OUT / "plots"
    pair = spectrogram_pair_2d(
        pre,
        wav,
        plots / "pair_sim_pre_post_2d.png",
        "模拟「术前」滤波（非患儿）",
        "TTS 原句示意「术后」（非患儿）",
        "模拟对照 · 非真实术前术后 · 非鼻音率",
    )
    pre3 = spectrogram_3d(pre, plots / "sim_pre_3d.png", "3D · 模拟术前滤波")
    post3 = spectrogram_3d(wav, plots / "sim_post_3d.png", "3D · TTS 示意术后")
    import json as _json

    sr_path = OUT / "json" / "say_card_self_ref.json"
    sr = _json.loads(sr_path.read_text(encoding="utf-8")) if sr_path.exists() else self_ref_from_rows([])
    pdf_path = OUT / "reports" / "阶段一-本机MVP预实验报告-模拟对照.pdf"
    write_pdf(
        pdf_path,
        pair,
        pre3,
        post3,
        sr,
        extra_lines=[
            f"2D 对照: {pair.name}",
            f"模拟术前 wav: {pre.name}",
            f"TTS 原句 wav: {wav.name}",
        ],
    )
    print(pair)
    print(pre3)
    print(post3)
    print(pdf_path)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    _banner()
    p = argparse.ArgumentParser(description="Local CLP hypernasality feature MVP")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="synthesize mono /a i u/ and extract features")
    sub.add_parser("demo-stereo", help="synthesize L=oral R=nasal pairs; print split_energy_index")
    a = sub.add_parser("analyze", help="analyze one local wav (mono or stereo, no upload)")
    a.add_argument("wav", type=Path)
    a.add_argument("--rater", default=None)
    a.add_argument("--grade", type=int, choices=[0, 1, 2, 3], default=None, help="hypernasality 0-3")
    a.add_argument("--notes", default=None)
    a.add_argument("--plot", action="store_true", help="Librosa 2D+3D spectrograms")
    g = sub.add_parser("align", help="SenseVoiceSmall timestamps + slice + Praat features")
    g.add_argument("wav", type=Path)
    g.add_argument("--keywords", default=None, help="comma-separated; 空则按朗读文本自动分目标/基线")
    g.add_argument("--prompt", default="", help="朗读文本，用于自参照与切片成功率")
    sub.add_parser("align-demo", help="macOS say 词表句 → align (not a patient sample)")
    sub.add_parser("report", help="模拟术前/术后图 + 自参照指数 + PDF（非患儿）")
    u = sub.add_parser("ui", help="本机网页界面（127.0.0.1）")
    u.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    if args.cmd == "demo":
        return cmd_demo()
    if args.cmd == "demo-stereo":
        return cmd_demo_stereo()
    if args.cmd == "align":
        return cmd_align(args.wav, args.keywords, args.prompt)
    if args.cmd == "align-demo":
        return cmd_align_demo()
    if args.cmd == "report":
        return cmd_report()
    if args.cmd == "ui":
        from .ui import launch

        launch(port=args.port)
        return 0
    return cmd_analyze(args.wav, args.rater, args.grade, args.notes, args.plot)


if __name__ == "__main__":
    raise SystemExit(main())
