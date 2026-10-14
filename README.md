# Nasal Acoustics MVP — Objective Hypernasality Feature Explorer for Cleft Lip & Palate (CLP)

> **Disclaimer:** Research and education aid only. This tool does **not** diagnose velopharyngeal insufficiency (VPI) or any condition, and its features are not a validated "nasalance" score. Perceptual ratings are entered by qualified clinicians. Do not upload identifiable patient data.

![Gradio UI (case list hidden for privacy)](docs/ui-screenshot.png)

## What it does
A local-only (127.0.0.1) app that extracts acoustic features related to hypernasality from short Mandarin speech prompts used in CLP speech follow-up:
- Synthetic vowel demos (/a i u/, low vs. high nasal stereo simulation)
- Word-level windowing of a prompt card sentence via SenseVoiceSmall (FunASR/ModelScope)
- Praat-based features (parselmouth), per-word JSON/CSV output, plots
- Self-referenced index and simulated pre/post-op comparison report (PDF)
- Gradio UI with pseudonymous case codes; data never leaves the machine

## Stack
Python 3, numpy, scipy, soundfile, matplotlib, praat-parselmouth, torch/torchaudio, funasr, modelscope, gradio.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nasal_mvp demo          # synthetic vowels -> output/
python -m nasal_mvp demo-stereo
python -m nasal_mvp ui --port 7860  # http://127.0.0.1:7860
# word alignment (downloads ~1GB model on first run)
export MODELSCOPE_CACHE="$PWD/.modelscope_cache"; export NASAL_MVP_DEVICE=cpu
python -m nasal_mvp align-demo
python -m nasal_mvp report
```

## Privacy & testing
The pipeline has been tested on real voluntary voice recordings made by the author. These recordings are **not published** for privacy reasons and are excluded from this repository. A synthetic audio generator (`python -m nasal_mvp demo`, `demo-stereo`) is included so anyone can test the pipeline without real recordings. Never commit clinical WAVs, names, or hospital IDs.

## Status
S1 MVP: the pipeline runs end-to-end on the author's real test recordings and on synthetic audio.

## Roadmap & collaboration
I am looking for **clinical speech-language pathologists (SLPs) and cleft speech-clinic collaborators**. Planned steps:
1. Ethics/IRB approval with partner clinics.
2. Collect paired pre-/post-operative recordings with informed consent, with clinician perceptual ratings of hypernasality.
3. Validate the acoustic features against those clinician ratings.
4. Goal: a **clinical speech-therapy assistance aid** that helps SLPs track progress. It is not a diagnostic device and will need clinical validation before any clinical use.

Interested? Please open an issue on this repository.

## Built with
Cursor, during MLH Global Hack Week: Hacktoberfest 2026.

## License
MIT — see LICENSE
