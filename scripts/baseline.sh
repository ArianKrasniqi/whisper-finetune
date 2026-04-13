#!/bin/bash
# Usage: ./scripts/baseline.sh <youtube_url> <output_name>
# Example: ./scripts/baseline.sh "https://www.youtube.com/..." "sample"

URL=$1
NAME=$2

if [ -z "$URL" ] || [ -z "$NAME" ]; then
  echo "Usage: ./scripts/baseline.sh <youtube_url> <output_name>"
  exit 1
fi

OUTPUT_DIR="samples/baseline"
mkdir -p "$OUTPUT_DIR"

echo "Downloading audio..."
yt-dlp -x --audio-format mp3 --audio-quality 0 -o "$OUTPUT_DIR/$NAME.%(ext)s" "$URL"

echo "Running Whisper transcription..."
whisper "$OUTPUT_DIR/$NAME.mp3" --model small --language sq --output_dir "$OUTPUT_DIR/"

echo "Done. Files saved in $OUTPUT_DIR/"
