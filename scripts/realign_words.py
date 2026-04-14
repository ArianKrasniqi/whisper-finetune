import argparse
import json
import whisperx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to aligned-adjusted.json")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading audio...")
    audio = whisperx.load_audio(args.audio)

    print("Loading alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code="sq",
        device="cpu",
        model_name="Alimzhan/wav2vec2-large-xls-r-300m-albanian-colab"
    )

    print("Re-aligning words with corrected text...")
    result = whisperx.align(data["segments"], model_a, metadata, audio, device="cpu", return_char_alignments=False)

    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Done. Updated {args.input}")
    print(f"Total segments: {len(result['segments'])}")

if __name__ == "__main__":
    main()
