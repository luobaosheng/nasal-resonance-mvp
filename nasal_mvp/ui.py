from __future__ import annotations

import json
from pathlib import Path

from . import DISCLAIMER, __version__
from .audio_in import resolve_audio
from .cases import (
    audio_path,
    code_from_choice,
    format_align_zh,
    format_features_zh,
    list_audio,
    save_audio,
    subject_choices,
    upsert_subject,
)
from .features import attach_perceptual, extract_features, write_json
from .plots import spectrogram_2d
from .prompts import add_prompt, load_bank

OUT = Path(__file__).resolve().parents[1] / "output"


def _need_subject(code: str, name: str, rater: str) -> str | None:
    if not (code or "").strip():
        return "请先填写编号，并点「保存/打开该被检人」。 / Enter a case code first, then click Save / Open case."
    upsert_subject(code, name, rater)
    return None


def _clip_choices(code: str) -> list[str]:
    return [row[0] for row in list_audio(code)]


def _dd(code: str):
    import gradio as gr

    ch = _clip_choices(code)
    return gr.update(choices=ch, value=(ch[0] if ch else None))


def ui_save_subject(code: str, name: str, rater: str):
    import gradio as gr

    if not (code or "").strip():
        ch = subject_choices()
        return gr.update(choices=ch), "编号不能空。 / Case code cannot be empty.", [], _dd("")
    upsert_subject(code, name, rater)
    ch = subject_choices()
    return gr.update(choices=ch, value=ch[0] if ch else None), f"已保存被检人 / Case saved: {code} {name}", list_audio(code), _dd(code)


def ui_pick_subject(choice: str):
    code = code_from_choice(choice)
    if not code:
        return "", "", "", [], _dd("")
    meta_p = OUT / "cases" / code / "meta.json"
    name, rater = "", ""
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        name = meta.get("name") or ""
        rater = meta.get("rater") or ""
    return code, name, rater, list_audio(code), _dd(code)


def _write_prompt(wav: Path, text: str) -> None:
    t = (text or "").strip()
    if t:
        wav.with_suffix(".txt").write_text(t, encoding="utf-8")


def ui_save_clip(audio, code: str, name: str, rater: str, label: str, prompt_text: str = ""):
    err = _need_subject(code, name, rater)
    if err:
        return None, err, list_audio(code) if code else [], _dd(code)
    src = resolve_audio(audio)
    if src is None:
        return None, "请先录音或导入音频。 / Record or upload audio first.", list_audio(code), _dd(code)
    dest = save_audio(code, src, label or "未标任务")
    _write_prompt(dest, prompt_text)
    return str(dest), f"已保存 / Saved: {dest.name}", list_audio(code), _dd(code)


def ui_use_listed(code: str, filename: str):
    if not code:
        return None, "请先打开被检人。 / Open a case first."
    if not filename:
        return None, "请在下拉框选择一段已保存录音。 / Choose a saved recording from the list."
    p = audio_path(code, str(filename))
    if p is None:
        return None, "找不到该文件。 / File not found."
    return str(p), f"已载入 / Loaded: {p.name}"


def ui_analyze(audio, code: str, name: str, rater: str, grade: str, label: str, prompt_text: str = ""):
    err = _need_subject(code, name, rater)
    if err:
        return None, err
    src = resolve_audio(audio)
    if src is None:
        return None, "请先录音、导入，或从列表载入一段。 / Record, upload, or load a saved recording first."
    dest = save_audio(code, src, label or "整段分析")
    _write_prompt(dest, prompt_text)
    rec = extract_features(dest)
    g = None
    if grade not in ("", "未填", None):
        try:
            g = int(str(grade))
        except ValueError:
            g = None
    rec = attach_perceptual(rec, rater=rater or None, hypernasality_0_3=g, notes=f"subject={code}")
    write_json(rec, OUT / "cases" / code / "analysis" / f"{dest.stem}.json")
    png = spectrogram_2d(dest, OUT / "cases" / code / "analysis" / f"{dest.stem}_spec2d.png", f"{code} {dest.name}")
    return str(png), format_features_zh(rec)


def ui_align(audio, code: str, name: str, rater: str, keywords: str, label: str, prompt_text: str = ""):
    from .__main__ import cmd_align

    err = _need_subject(code, name, rater)
    if err:
        return err
    src = resolve_audio(audio)
    if src is None:
        return "请先录音、导入，或从列表载入一段。 / Record, upload, or load a saved recording first."
    dest = save_audio(code, src, label or "切窗")
    _write_prompt(dest, prompt_text)
    cmd_align(dest, (keywords or "").strip() or None, prompt_text)
    sr_path = OUT / "json" / f"{dest.stem}_self_ref.json"
    extra = {}
    if sr_path.exists():
        extra = json.loads(sr_path.read_text(encoding="utf-8"))
        (OUT / "cases" / code / "analysis" / f"{dest.stem}_self_ref.json").write_text(
            json.dumps(extra, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    csv_p = OUT / f"features_align_{dest.stem}.csv"
    tail = csv_p.read_text(encoding="utf-8") if csv_p.exists() else ""
    return format_align_zh(extra, tail)


def ui_tts() -> str:
    from .__main__ import cmd_align_demo

    cmd_align_demo()
    sr = OUT / "json" / "say_card_self_ref.json"
    extra = json.loads(sr.read_text(encoding="utf-8")) if sr.exists() else {}
    csv_p = OUT / "features_align_say_card.csv"
    tail = csv_p.read_text(encoding="utf-8") if csv_p.exists() else ""
    return format_align_zh(extra, tail, note="TTS 演示句，非患儿。 / TTS demo sentence, not a patient.")


def ui_report() -> tuple:
    from .__main__ import cmd_report

    cmd_report()
    png = OUT / "plots" / "pair_sim_pre_post_2d.png"
    pdf = OUT / "reports" / "阶段一-本机MVP预实验报告-模拟对照.pdf"
    note = f"PDF：{pdf}\n对照为 DSP 模拟，非真实患儿。 / Simulated comparison, not real patients."
    return (str(png) if png.exists() else None), note


def launch(port: int = 7860, inbrowser: bool = True) -> None:
    import gradio as gr

    css = """
    #feat-left textarea, #feat-right textarea {
      max-height: 28rem !important;
      overflow-y: auto !important;
    }
    """
    with gr.Blocks(title="鼻音声学 MVP / Nasal Resonance MVP", css=css) as demo:
        gr.Markdown(
            f"# 唇腭裂过高鼻音客观化 · 本机 MVP / Cleft Palate Hypernasality — Local MVP `{__version__}`\n\n"
            f"{DISCLAIMER} 仅 127.0.0.1。编号请用内部号，勿用住院号；本机保存，不上云。\n\n*Research and education aid only, not a diagnostic tool. Runs on 127.0.0.1 only; use internal case codes, never hospital IDs; data stays on this computer.*"
        )
        with gr.Row():
            code = gr.Textbox(label="编号（内部） / Case code (internal)", scale=1)
            subj = gr.Textbox(label="被检人姓名 / Speaker name", scale=1)
            rater = gr.Textbox(label="听感评分人 / Perceptual rater", scale=1)
        with gr.Row():
            btn_save = gr.Button("保存/打开该被检人 / Save / Open case", variant="primary")
            picker = gr.Dropdown(label="已有被检人 / Existing cases", choices=subject_choices(), allow_custom_value=False)
        files = gr.Dataframe(
            headers=["文件名 / File", "任务标签 / Task", "路径 / Path"],
            label="该编号下已保存的录音 / Saved recordings for this case",
            interactive=False,
            wrap=True,
        )
        clip_dd = gr.Dropdown(label="选择已保存的一段 / Choose a saved recording", choices=[])
        load_btn = gr.Button("载入到下面的共用音频 / Load into shared audio below")
        status = gr.Textbox(label="状态 / Status", lines=2)

        gr.Markdown("### 朗读提示（请被检人照此朗读后再录音） / Reading prompt (speaker reads this, then record)")
        with gr.Row():
            prompt_dd = gr.Dropdown(label="常用测试句 / Saved test sentences", choices=load_bank(), value=load_bank()[0])
            btn_load_p = gr.Button("载入到朗读框 / Load into prompt")
            btn_add_p = gr.Button("把当前句加入常用 / Save current sentence")
        prompt = gr.Textbox(
            label="请照此朗读 / Please read aloud",
            value=load_bank()[0],
            lines=3,
            placeholder="可改字、可从常用句载入 / Edit, or load a saved sentence",
        )

        gr.Markdown("### 共用音频（整段分析与 SenseVoice 切窗用同一段） / Shared audio (used by both whole-clip analysis and SenseVoice windows)")
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="录音或导入 wav / Record or upload audio (wav)")
        label = gr.Textbox(label="本段任务标签（如：卡片句 / 持续/a/ / 术后2周） / Task label (e.g. card sentence, sustained /a/, 2 weeks post-op)", value="卡片句")
        save_clip = gr.Button("仅保存本段到该编号（不分析） / Save to case only (no analysis)")

        with gr.Row():
            with gr.Column():
                grade = gr.Dropdown(["未填", "0", "1", "2", "3"], value="未填", label="过高鼻音 0–3（人工） / Hypernasality 0–3 (clinician rating)")
                btn_an = gr.Button("整段分析 + 2D 语谱 / Analyze whole clip + spectrogram")
            with gr.Column():
                kws = gr.Textbox(label="切窗关键词 / Window keywords", value="爸,怕,打,靶,妈,奶")
                btn_al = gr.Button("同一段：SenseVoice 切窗 + 自参照 / SenseVoice word windows + self-ref index")
        img = gr.Image(label="2D 语谱图 / Spectrogram")
        with gr.Row():
            feat_left = gr.Textbox(
                label="左：整段分析特征 / Left: whole-clip features",
                lines=18,
                max_lines=18,
                autoscroll=False,
                elem_id="feat-left",
            )
            feat_right = gr.Textbox(
                label="右：SenseVoice 切窗 + 自参照 / Right: word windows + self-ref index",
                lines=18,
                max_lines=18,
                autoscroll=False,
                elem_id="feat-right",
            )
        with gr.Row():
            gr.Button("TTS 演示句（非患儿） / TTS demo (not a patient)").click(ui_tts, outputs=feat_right)
            btn_rep = gr.Button("生成模拟术前/术后 PDF / Generate simulated pre/post PDF")
        img_rep = gr.Image(label="模拟对照图 / Simulated comparison chart")
        note_rep = gr.Textbox(label="报告路径 / Report path")

        def _load_prompt(choice):
            return choice or ""

        def _add_prompt(text):
            import gradio as gr

            bank = add_prompt(text)
            return gr.update(choices=bank, value=text.strip() if text and text.strip() else None), "已加入常用测试句。 / Added to saved sentences."

        btn_load_p.click(_load_prompt, [prompt_dd], [prompt])
        prompt_dd.change(_load_prompt, [prompt_dd], [prompt])
        btn_add_p.click(_add_prompt, [prompt], [prompt_dd, status])

        btn_save.click(ui_save_subject, [code, subj, rater], [picker, status, files, clip_dd])
        picker.change(ui_pick_subject, [picker], [code, subj, rater, files, clip_dd])
        save_clip.click(ui_save_clip, [audio, code, subj, rater, label, prompt], [audio, status, files, clip_dd])
        load_btn.click(ui_use_listed, [code, clip_dd], [audio, status])

        def _refresh_lists(c):
            return list_audio(c), _dd(c)

        btn_an.click(ui_analyze, [audio, code, subj, rater, grade, label, prompt], [img, feat_left]).then(
            _refresh_lists, [code], [files, clip_dd]
        )
        btn_al.click(ui_align, [audio, code, subj, rater, kws, label, prompt], [feat_right]).then(
            _refresh_lists, [code], [files, clip_dd]
        )
        btn_rep.click(ui_report, outputs=[img_rep, note_rep])

    demo.launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=inbrowser)
