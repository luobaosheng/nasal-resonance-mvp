# Nasal Resonance MVP — Objective Acoustic Features for Hypernasality in Cleft Palate Speech

![Gradio UI (case list hidden for privacy)](docs/ui-screenshot.png)

> **Disclaimer:** Research and education aid only. This tool does **not** diagnose velopharyngeal insufficiency (VPI), does not replace nasendoscopy, a Nasometer, or a speech-language pathologist's perceptual rating, and its single-microphone features are **not** a nasalance score. Do not upload identifiable patient data.

## The clinical problem
Children born with cleft lip and palate (CLP) often have hypernasal speech caused by VPI and may need years of surgery and speech therapy. Progress is judged mainly by ear: perceptual ratings are subjective and hard to compare across visits and clinicians. Dual-channel nasometry is expensive and not available everywhere, and nasendoscopy is invasive and frightening for children. Earlier phone-based attempts struggled with automatic gain control, whole-utterance averaging, and absolute values that change from device to device.

## Who it is for
- Speech-language pathologists (SLPs) and cleft speech teams who want an objective, repeatable record **alongside** their perceptual rating
- Researchers studying acoustic markers of hypernasality in Mandarin-speaking children

The clinician's ear stays primary; the tool is a secondary, objective record.

## Method
1. **Known-text prompt cards:** the speaker reads a standard Mandarin sentence that mixes oral and nasal-sensitive sounds (e.g. 爸 / 怕 / 打 / 妈 / 奶).
2. **Single-channel recording** (a lavalier mic with AGC off is recommended for consistency).
3. **Word-level windows:** SenseVoiceSmall (FunASR) aligns the known text and cuts a window around each target word. You can also adjust windows by hand if the alignment is wrong.
4. **Acoustic features per window** with Praat (parselmouth): formants, spectral energy measures and spectrograms. High-frequency (2–4 kHz) slope is reported separately, because it relates more to nasal air emission than to hypernasality.
5. **Self-referenced index:** a dimensionless ratio between pressure-consonant windows and oral-vowel windows *within the same recording*. It reduces device and gain effects compared with absolute values. This is a research index, not a nasalance score.
6. **Output:** per-word JSON/CSV, plots, multi-recording case history and a pre/post comparison PDF. A clinician's 0–3 perceptual grade can be stored next to the acoustic features for later comparison.

Everything runs locally on 127.0.0.1. Audio never leaves the computer.

## Stack
Python 3, numpy, scipy, soundfile, matplotlib, praat-parselmouth, torch/torchaudio, funasr, modelscope, gradio.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nasal_mvp demo            # synthetic vowels /a i u/ -> output/
python -m nasal_mvp demo-stereo     # synthetic low vs high nasal (dual-channel upgrade smoke test)
python -m nasal_mvp ui --port 7860  # http://127.0.0.1:7860
export MODELSCOPE_CACHE="$PWD/.modelscope_cache"; export NASAL_MVP_DEVICE=cpu
python -m nasal_mvp align-demo      # word alignment (downloads ~1 GB model on first run)
python -m nasal_mvp analyze /path/to/file.wav --plot
python -m nasal_mvp report          # simulated pre/post report PDF
```

## Privacy & testing
The pipeline has been tested end-to-end on real voluntary voice recordings made by the author. Those recordings are **not published** for privacy reasons and are not in this repository. A synthetic audio generator is included so anyone can test the pipeline without real recordings. No child or patient recordings have been collected; none will be until ethics approval and informed consent are in place. Never commit clinical WAVs, names, or hospital IDs.

## Current stage
**Stage 1 (single channel) — engineering proof of concept.** Working: recording → spectrogram → word-level windows → self-referenced index → case history → report. Not yet done: an agreed prompt-card word list and rating scale with clinicians, a frozen recording protocol, ethics approval, and any validation against clinician ratings. Pediatric speech alignment has not been tested yet.

## Roadmap & collaboration
I am looking for **clinical speech-language pathologists and cleft speech-clinic collaborators**.
1. Agree on the prompt-card word list, the 0–3 perceptual scale and the recording protocol with SLPs.
2. Obtain ethics/IRB approval with a partner clinic (child recordings with informed consent, minimal pseudonymous data).
3. Collect a small set of paired pre-/post-operative recordings with clinician perceptual ratings.
4. Compare the acoustic features with the clinicians' ratings; publish the method openly.
5. Stage 2 (only if justified): dual-channel oral/nasal recording closer to nasometry.
6. Long-term goal: a **speech-therapy assistance aid** that helps SLPs track progress. It is not a diagnostic device and needs clinical validation before any clinical use.

Interested in collaborating? Please open an issue on this repository.

## Built with
Cursor, during MLH Global Hack Week: Hacktoberfest 2026.

## License
MIT — see LICENSE
