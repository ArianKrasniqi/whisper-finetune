import argparse
import json
import os
import subprocess
from datasets import Dataset, Audio

def slice_audio(audio_path, start, end, output_path):
    """Use ffmpeg to slice a chunk from the audio file."""
    subprocess.run([
        "/usr/local/bin/ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-to", str(end),
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True, help="Path to aligned-adjusted.json")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--output_dir", required=True, help="Where to save the HuggingFace dataset")
    parser.add_argument("--chunks_dir", default=None, help="Optional: folder to save mp3 chunks for review")
    args = parser.parse_args()

    with open(args.aligned, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

    wav_dir = os.path.join(args.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    if args.chunks_dir:
        os.makedirs(args.chunks_dir, exist_ok=True)

    wav_paths = []
    texts = []

    print(f"Processing {len(segments)} segments...")

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()

        # Slice and save as wav (referenced by path in the dataset)
        wav_path = os.path.join(wav_dir, f"chunk_{i:03d}.wav")
        slice_audio(args.audio, start, end, wav_path)

        wav_paths.append(wav_path)
        texts.append(text)

        # Optionally save as mp3 chunk for review
        if args.chunks_dir:
            chunk_mp3 = os.path.join(args.chunks_dir, f"chunk_{i:03d}_{start:.1f}-{end:.1f}s.mp3")
            slice_audio(args.audio, start, end, chunk_mp3)

        print(f"  [{i+1}/{len(segments)}] {start:.2f}s - {end:.2f}s: {text[:50]}")

    print("Building HuggingFace dataset...")
    dataset = Dataset.from_dict({
        "audio": wav_paths,
        "text": texts
    })
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(args.output_dir)

    print(f"\nDone. Dataset saved to {args.output_dir}")
    print(f"Total examples: {len(dataset)}")
    if args.chunks_dir:
        print(f"Review chunks saved to {args.chunks_dir}")

if __name__ == "__main__":
    main()
