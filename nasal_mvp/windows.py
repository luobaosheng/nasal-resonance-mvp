from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def audio_duration_ms(src: Path) -> float:
    info = sf.info(str(src))
    return 1000.0 * info.frames / info.samplerate


def slice_wav(
    src: Path,
    dest: Path,
    start_ms: float,
    end_ms: float,
    pad_ms: float = 40.0,
    max_dur_ms: float | None = None,
) -> Path:
    data, sr = sf.read(str(src), always_2d=True)
    n = data.shape[0]
    dur_ms = 1000.0 * n / sr
    if max_dur_ms is not None:
        end_ms = min(end_ms, start_ms + max_dur_ms)
    if start_ms >= dur_ms or end_ms <= 0 or end_ms <= start_ms:
        raise ValueError(f"window outside audio {start_ms}-{end_ms} ms (dur {dur_ms:.0f})")
    a = max(int((start_ms - pad_ms) / 1000.0 * sr), 0)
    b = min(int((end_ms + pad_ms) / 1000.0 * sr), n)
    if b - a < int(0.02 * sr):
        raise ValueError(f"empty window {start_ms}-{end_ms} ms")
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), data[a:b], sr, subtype="PCM_16")
    return dest


def write_mono16k_from_float(path: Path, samples: np.ndarray, sr: int) -> Path:
    """Resample-free write; caller should already be near 16 kHz if feeding ASR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path
