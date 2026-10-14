from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["font.sans-serif"] = ["PingFang SC", "PingFang HK", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def _mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    return y, sr


def spectrogram_2d(path: Path, dest: Path, title: str) -> Path:
    y, sr = _mono(path)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_ylim(0, 5000)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.text(0.01, 0.01, "单通道语谱图 · 研究示意 · 非鼻音率 · 非诊断", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def spectrogram_3d(path: Path, dest: Path, title: str) -> Path:
    y, sr = _mono(path)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    # downsample for surface
    S_db = S_db[::4, ::2]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)[::4]
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=256)[::2]
    T, F = np.meshgrid(times, freqs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, F, S_db, cmap="magma", linewidth=0, antialiased=False, rcount=80, ccount=80)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_zlabel("dB")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 5000)
    fig.text(0.01, 0.01, "3D 瀑布图为展示用 · 非临床金标准", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def envelope_overlay(target_wavs: list[Path], baseline_wavs: list[Path], dest: Path) -> Path | None:
    if not target_wavs or not baseline_wavs:
        return None

    def env(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        acc = None
        sr0 = 16000
        for p in paths[:6]:
            y, sr = _mono(p)
            sr0 = sr
            spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            if acc is None:
                acc = spec
            else:
                n = min(len(acc), len(spec))
                acc = acc[:n] + spec[:n]
        acc = acc / max(len(paths[:6]), 1)
        freqs = np.fft.rfftfreq(2 * (len(acc) - 1), 1.0 / sr0)
        db = 20 * np.log10(acc + 1e-12)
        return freqs, db

    ft, et = env(target_wavs)
    fb, eb = env(baseline_wavs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ft, et, label="口腔目标窗", color="#c0392b")
    ax.plot(fb, eb, label="鼻音基线窗", color="#2980b9")
    ax.set_xlim(0, 5000)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("相对能量 (dB)")
    ax.set_title("目标窗 vs 基线窗 频谱包络（同一次录音自参照）")
    ax.legend()
    fig.text(0.01, 0.01, "非 Nasometer · 非诊断", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def spectrogram_pair_2d(path_a: Path, path_b: Path, dest: Path, title_a: str, title_b: str, banner: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, p, t in ((axes[0], path_a, title_a), (axes[1], path_b, title_b)):
        y, sr = _mono(p)
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_ylim(0, 5000)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("时间 (s)")
    axes[0].set_ylabel("频率 (Hz)")
    fig.colorbar(img, ax=axes, format="%+2.0f dB", shrink=0.8)
    fig.suptitle(banner, fontsize=9, color="0.25")
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest
