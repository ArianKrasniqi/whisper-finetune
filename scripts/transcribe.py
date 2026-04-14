import argparse
import torch
import torchaudio
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model", default="models/whisper-small-albanian", help="Path to model")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()

    print("Loading audio...")
    audio_array, sample_rate = sf.read(args.audio)
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sample_rate != 16000:
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
        audio_array = audio_tensor.squeeze(0).numpy()

    inputs = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")

    print("Transcribing...")
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features,
            language="albanian",
            task="transcribe",
        )

    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print("\n--- Transcription ---")
    print(transcription)

if __name__ == "__main__":
    main()
