from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

from . import SCHEMA


def _median_nonzero(arr: np.ndarray) -> float | None:
    v = arr[np.isfinite(arr) & (arr > 0)]
    if v.size == 0:
        return None
    return float(np.median(v))


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (n_samples, n_channels) float64 and sample rate."""
    data, sr = sf.read(str(path), always_2d=True)
    return data.astype(np.float64), int(sr)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def split_energy_index(oral: np.ndarray, nasal: np.ndarray) -> float:
    """Research index 0-100: nasal / (oral + nasal). Not Nasometer nasalance."""
    o, n = _rms(oral), _rms(nasal)
    return round(100.0 * n / (o + n), 3)


def extract_features(wav_path: Path) -> dict[str, Any]:
    wav_path = Path(wav_path)
    data, sr = load_wav(wav_path)
    n_ch = data.shape[1]
    # Convention: stereo L = oral/larynx, R = nasal. Mono = mix (no split index).
    if n_ch >= 2:
        oral, nasal = data[:, 0], data[:, 1]
        samples = oral.copy()
        channel_label = "stereo_L_oral_or_larynx_R_nasal"
        split_idx = split_energy_index(oral, nasal)
        oral_rms, nasal_rms = _rms(oral), _rms(nasal)
    else:
        samples = data[:, 0]
        channel_label = "mono"
        split_idx = None
        oral_rms, nasal_rms = None, None

    peak = np.max(np.abs(samples)) or 1.0
    if peak > 1.0:
        samples = samples / peak
    duration = float(len(samples) / sr)
    snd = parselmouth.Sound(samples, sampling_frequency=sr)

    pitch = snd.to_pitch()
    f0 = _median_nonzero(pitch.selected_array["frequency"])

    formant = snd.to_formant_burg()
    n = formant.get_number_of_frames()
    f1s, f2s, f3s, bw1 = [], [], [], []
    for i in range(1, n + 1):
        t = formant.get_time_from_frame_number(i)
        for bucket, idx in ((f1s, 1), (f2s, 2), (f3s, 3)):
            val = formant.get_value_at_time(idx, t)
            if val and np.isfinite(val):
                bucket.append(val)
        b1 = formant.get_bandwidth_at_time(1, t)
        if b1 and np.isfinite(b1) and b1 > 0:
            bw1.append(b1)

    harmonicity = snd.to_harmonicity()
    hnr = call(harmonicity, "Get mean", 0, 0)

    spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / sr)
    power = spec ** 2
    total = float(power.sum()) or 1.0
    centroid = float((freqs * power).sum() / total)
    low_mask = freqs < 1000
    low_ratio = float(power[low_mask].sum() / total)
    hf_mask = (freqs >= 2000) & (freqs <= 4000)
    hf_slope = None
    hf_db_oct = None
    if int(hf_mask.sum()) >= 8:
        coef = np.polyfit(freqs[hf_mask], np.log10(power[hf_mask] + 1e-20), 1)
        hf_slope = round(float(coef[0]) * 1000.0, 6)
        band2 = (freqs >= 1800) & (freqs <= 2200)
        band4 = (freqs >= 3800) & (freqs <= 4200)
        if int(band2.sum()) and int(band4.sum()):
            p2 = float(np.mean(power[band2]) + 1e-20)
            p4 = float(np.mean(power[band4]) + 1e-20)
            hf_db_oct = round(10.0 * np.log10(p4 / p2), 2)  # 2 kHz→4 kHz 一个倍频
    rms_energy = float(np.mean(np.square(samples)))

    return {
        "schema": SCHEMA,
        "source_file": wav_path.name,
        "sample_rate_hz": sr,
        "duration_s": round(duration, 3),
        "n_channels_analyzed": n_ch,
        "features": {
            "f0_median_hz": None if f0 is None else round(f0, 2),
            "f1_median_hz": None if not f1s else round(float(np.median(f1s)), 2),
            "f1_bandwidth_hz": None if not bw1 else round(float(np.median(bw1)), 2),
            "f2_median_hz": None if not f2s else round(float(np.median(f2s)), 2),
            "f3_median_hz": None if not f3s else round(float(np.median(f3s)), 2),
            "hnr_mean_db": None if not np.isfinite(hnr) else round(float(hnr), 2),
            "spectral_centroid_hz": round(centroid, 2),
            "energy_below_1khz_ratio": round(low_ratio, 4),
            "hf_slope_2_4khz": hf_slope,
            "hf_db_per_octave_2k_4k": hf_db_oct,
            "rms_energy": round(rms_energy, 8),
            "oral_or_larynx_rms": None if oral_rms is None else round(oral_rms, 6),
            "nasal_rms": None if nasal_rms is None else round(nasal_rms, 6),
            "split_energy_index": split_idx,
        },
        "labels": {
            "not_nasalance_percent": True,
            "channel": channel_label,
            "split_energy_index_is_not_nasometer": True,
        },
        "perceptual": {
            "rater": None,
            "hypernasality_0_3": None,
            "nasal_emission": None,
            "quality_ok": None,
            "notes": None,
        },
    }


def attach_perceptual(rec: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    p = rec.setdefault("perceptual", {})
    for k, v in kwargs.items():
        if v is not None:
            p[k] = v
    return rec


def write_json(rec: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")


def estimate_snr_db(wav_path: Path) -> float | None:
    data, sr = load_wav(wav_path)
    y = data.mean(axis=1)
    n = int(0.08 * sr)
    if n < 8 or len(y) < 3 * n:
        return None
    noise = np.concatenate([y[:n], y[-n:]])
    sig = y[n:-n]
    rn, rs = _rms(noise), _rms(sig)
    if rn <= 0:
        return None
    return round(20.0 * np.log10(rs / rn), 1)
