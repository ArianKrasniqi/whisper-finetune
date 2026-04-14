#!/bin/bash
# Usage: ./scripts/align.sh <audio_file> <transcript_file> <output_dir>
# Example: ./scripts/align.sh data/video1/audio.mp3 data/video1/transcript.txt data/video1/

AUDIO=$1
TRANSCRIPT=$2
OUTPUT_DIR=$3

if [ -z "$AUDIO" ] || [ -z "$TRANSCRIPT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: ./scripts/align.sh <audio_file> <transcript_file> <output_dir>"
  exit 1
fi

source venv/bin/activate

python scripts/align.py --audio "$AUDIO" --transcript "$TRANSCRIPT" --output_dir "$OUTPUT_DIR"
