"""Synthetic vowels for pipeline checks only — not clinical gold standard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

# Approximate adult male F1/F2 (Hz)
VOWELS = {
    "a": (730, 1090, 2440),
    "i": (270, 2290, 3010),
    "u": (300, 870, 2240),
}


def synth_vowel(name: str, duration: float = 1.2, sr: int = 44100, f0: float = 120.0) -> np.ndarray:
    f1, f2, f3 = VOWELS[name]
    t = np.arange(int(duration * sr)) / sr
    # Pulse train-ish: sum of harmonics with formant envelope via resonators (simple IIR)
    pulse = np.zeros_like(t)
    period = int(sr / f0)
    pulse[::period] = 1.0
    y = pulse.astype(np.float64)
    for fc, bw in ((f1, 90.0), (f2, 110.0), (f3, 140.0)):
        y = _resonator(y, fc, bw, sr)
    # fade
    fade = int(0.02 * sr)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    y /= np.max(np.abs(y)) or 1.0
    y *= 0.3
    return y.astype(np.float32)


def _resonator(x: np.ndarray, fc: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * fc / sr
    a1 = 2 * r * np.cos(theta)
    a2 = -(r ** 2)
    y = np.zeros_like(x)
    for n in range(2, len(x)):
        y[n] = x[n] + a1 * y[n - 1] + a2 * y[n - 2]
    return y


def write_demo_wavs(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("a", "i", "u"):
        p = out_dir / f"synth_vowel_{name}.wav"
        sf.write(str(p), synth_vowel(name), 44100, subtype="PCM_16")
        paths.append(p)
    return paths


def write_demo_stereo(out_dir: Path) -> list[Path]:
    """L=oral/larynx, R=nasal. Two known mix ratios for formula check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    oral = synth_vowel("a")
    paths = []
    for tag, nasal_gain in (("low_nasal", 0.15), ("high_nasal", 0.85)):
        nasal = oral * nasal_gain
        stereo = np.stack([oral, nasal], axis=1)
        p = out_dir / f"synth_stereo_a_{tag}.wav"
        sf.write(str(p), stereo, 44100, subtype="PCM_16")
        paths.append(p)
    return paths
