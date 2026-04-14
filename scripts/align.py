import argparse
import json
import os
import whisperx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading audio...")
    audio = whisperx.load_audio(args.audio)

    # Step 1: Run Whisper transcription to get rough segments with timestamps
    print("Transcribing with Whisper (to get initial timestamps)...")
    whisper_model = whisperx.load_model("small", device="cpu", language="sq")
    result = whisper_model.transcribe(audio, language="sq")

    # Step 2: Load Albanian alignment model and refine the timestamps
    print("Loading alignment model...")
    # Albanian has no default model in WhisperX, so we specify one explicitly
    model_a, metadata = whisperx.load_align_model(
        language_code="sq",
        device="cpu",
        model_name="Alimzhan/wav2vec2-large-xls-r-300m-albanian-colab"
    )

    print("Aligning...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu", return_char_alignments=False)

    output_path = os.path.join(args.output_dir, "aligned.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to {output_path}")
    print(f"Total segments: {len(result['segments'])}")

if __name__ == "__main__":
    main()
