# Albanian Speech-to-Text

Fine-tuning OpenAI's Whisper model on Albanian audio from a specific speaker. The model improves over time as new videos and transcripts are added.

## Setup

```bash
source venv/bin/activate
```

## Scripts

### `scripts/baseline.sh` — Download + transcribe a video

Downloads audio from a YouTube URL and runs a baseline Whisper transcription on it.

```bash
chmod +x scripts/baseline.sh  # only needed once
./scripts/baseline.sh "<youtube_url>" "<output_name>"
```

Example:
```bash
./scripts/baseline.sh "https://www.youtube.com/shorts/4arPW5Qc9DI" "sample"
```

Output files are saved in `samples/baseline/` as `.mp3`, `.txt`, `.srt`, `.vtt`, `.tsv`, and `.json`.

## See Also

- `PLAN.md` — full 7-step project plan and progress tracker
