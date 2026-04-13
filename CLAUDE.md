# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Fine-tune OpenAI's `whisper-small` model on Albanian audio from a single speaker. A new speaker will be added in the near future, when good results are achieved with the first one. The model improves incrementally — each week, new videos + correct transcripts are added and the model is retrained on all accumulated data.

## Environment Setup

- Python 3.11.9 (pinned via `.python-version`)
- Activate the venv before running anything: `source venv/bin/activate`
- PyTorch MPS backend (Apple Silicon M3) — use `device="mps"` for training and inference
- ffmpeg is installed at `/usr/local/bin/ffmpeg`

## Pipeline (in order)

```
YouTube URL → yt-dlp → audio (.mp3)
audio + correct transcript → WhisperX (forced alignment) → timestamped segments (.json)
all segments → HuggingFace dataset → fine-tune whisper-small → model checkpoint
model checkpoint + new audio → transcription
```

## Directory Structure

```
samples/baseline/   # Base Whisper output before any fine-tuning (for quality comparison)
samples/v1/         # Output after first fine-tune round (future)
data/               # Training material: audio files + correct Albanian transcripts
models/             # Saved model checkpoints (future)
```

## Key Decisions

- **No manual timestamps** — WhisperX forced alignment auto-generates them from the correct transcript text
- **Retrain on all data** each round (not just new files) — this is intentional for cumulative improvement
- **whisper-small** — chosen for feasibility on M3 overnight training; do not switch to a larger model without user confirmation
- **Language: Albanian (sq)** — always pass `language="sq"` to Whisper/WhisperX calls

## Current Progress

See `PLAN.md` for the 7-step plan and completion status. Steps 1 and 2 are done.
