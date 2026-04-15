# Albanian Speech-to-Text

Fine-tuning OpenAI's Whisper model on Albanian audio from a specific speaker. The model improves over time as new videos and transcripts are added.

## Setup

```bash
source venv/bin/activate
```

## Scripts

### `scripts/baseline.sh` — Download + transcribe a video

```bash
chmod +x scripts/baseline.sh
./scripts/baseline.sh "<youtube_url>" "<output_name>"
```

---

### `scripts/align.py` — Forced alignment

```bash
python scripts/align.py --audio data/video1/audio.mp3 --output_dir data/video1/
```

---

### `scripts/realign_words.py` — Re-align words after text correction

```bash
python scripts/realign_words.py --input data/video1/aligned-adjusted.json --audio data/video1/audio.mp3
```

---

### `scripts/build_dataset.py` — Build training dataset

```bash
python scripts/build_dataset.py --aligned data/video1/aligned-adjusted.json --audio data/video1/audio.mp3 --output_dir dataset/ --chunks_dir data/video1/chunks/
```

---

### `scripts/train.py` — Fine-tune whisper-small

```bash
python scripts/train.py
```

---

### `scripts/transcribe.py` — Transcribe audio with fine-tuned model

```bash
python scripts/transcribe.py --audio data/video1/audio.mp3 --model models/whisper-small-albanian
```

---

## Adding New Training Data

Each new video goes in its own folder:

```
data/
  video1/
  video2/
  video3/
```

For each new video:
1. Download audio with `baseline.sh`
2. Run `align.py`
3. Correct text → save as `aligned-adjusted.json`
4. Run `realign_words.py`
5. Listen to chunks, remove bad segments
6. Rebuild dataset with `build_dataset.py`
7. Retrain with `train.py`
