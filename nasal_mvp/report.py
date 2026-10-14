from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

DISCLAIMER = (
    "研究用预实验报告。单通道自参照指数 ≠ Nasometer 鼻音率。"
    "「术前/术后」对照为 DSP 模拟，不是真实患儿。"
    "不替代语音治疗师听感，不诊断腭咽闭合不全。"
)


def _page_text(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.93, title, fontsize=14, weight="bold", va="top")
    y = 0.86
    for line in lines:
        fig.text(0.08, y, line, fontsize=10, va="top", wrap=True)
        y -= 0.038
    fig.text(0.08, 0.04, DISCLAIMER, fontsize=7.5, color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


def _page_image(pdf: PdfPages, img: Path, caption: str) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.80])
    ax.imshow(plt.imread(str(img)))
    ax.axis("off")
    fig.text(0.06, 0.05, caption, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(
    dest: Path,
    pair_png: Path,
    pre_3d: Path,
    post_3d: Path,
    self_ref: dict,
    extra_lines: list[str],
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(dest) as pdf:
        _page_text(
            pdf,
            "阶段一预实验报告（本机 MVP）",
            [
                "管道：卡片句 → SenseVoiceSmall 字级切窗 → Praat 共振峰 + Librosa STFT 图。",
                "指数：self_ref_index = 口高压字窗(低频能量比) / 妈奶窗(低频能量比)。",
                "hf_slope_2_4kHz 单列，不与过高鼻音混成一个分。",
                "",
                f"self_ref_index = {self_ref.get('self_ref_index')}  （not_nasalance=True）",
                f"口高压窗 n={self_ref.get('n_oral_windows')}  妈奶窗 n={self_ref.get('n_nasal_char_windows')}",
                f"oral_low={self_ref.get('oral_low_ratio_mean')}  nasal_char_low={self_ref.get('nasal_char_low_ratio_mean')}",
                "",
                *extra_lines,
                "",
                "缺：伦理批件、真实术前术后成对录音、临床评估者听感金标准。",
                "本 PDF 仅证明本机管道与作图可复现。",
            ],
        )
        _page_image(pdf, pair_png, "图1  2D 语谱对照（左=模拟术前滤波，右=TTS 原句作「术后」示意）")
        _page_image(pdf, pre_3d, "图2  3D 瀑布图 · 模拟术前（展示用）")
        _page_image(pdf, post_3d, "图3  3D 瀑布图 · TTS 原句示意术后（展示用）")
        d = pdf.infodict()
        d["Title"] = "Nasal acoustics MVP pre-experiment (simulated)"
        d["Author"] = "Foson local MVP"
        d["Subject"] = "Not a diagnostic device; not Nasometer nasalance"
    return dest
