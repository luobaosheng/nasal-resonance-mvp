"""DSP stand-in for 术前/术后. NOT a patient recording."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def simulate_hypernasal(src: Path, dest: Path) -> Path:
    """Boost 250–800 Hz, attenuate 2–4 kHz — labeled 模拟过高鼻音."""
    data, sr = sf.read(str(src), always_2d=True)
    x = data.mean(axis=1)
    ny = 0.5 * sr
    sos_low = butter(2, [250 / ny, 800 / ny], btype="band", output="sos")
    sos_hi = butter(2, [2000 / ny, min(4000 / ny, 0.95)], btype="band", output="sos")
    nasalish = x + 0.85 * sosfilt(sos_low, x) - 0.55 * sosfilt(sos_hi, x)
    peak = np.max(np.abs(nasalish)) or 1.0
    nasalish = 0.9 * nasalish / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), nasalish.astype(np.float32), sr, subtype="PCM_16")
    return dest
