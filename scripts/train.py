import torch
import torchaudio
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset/"
MODEL_NAME  = "openai/whisper-small"
OUTPUT_DIR  = "models/whisper-small-albanian"
LANGUAGE    = "albanian"
TASK        = "transcribe"

# ── Load processor (feature extractor + tokenizer) ────────────────────────────

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# ── Load dataset ───────────────────────────────────────────────────────────────

print("Loading dataset...")
dataset = load_from_disk(DATASET_DIR)
print(f"Total examples: {len(dataset)}")

# Split into train (90%) and eval (10%)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Prepare each example for Whisper ──────────────────────────────────────────

def prepare_example(example):
    # Load audio directly from wav file to avoid torchcodec dependency
    audio_array, sample_rate = sf.read(example["audio"])
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    # Resample to 16000 Hz if needed (Whisper requires 16kHz)
    if sample_rate != 16000:
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
        audio_array = audio_tensor.squeeze(0).numpy()
    example["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=16000
    ).input_features[0]
    example["labels"] = processor.tokenizer(example["text"]).input_ids
    return example

print("Preparing dataset...")
train_dataset = train_dataset.map(prepare_example, remove_columns=["audio", "text"])
eval_dataset  = eval_dataset.map(prepare_example, remove_columns=["audio", "text"])

# ── Data collator ─────────────────────────────────────────────────────────────
# Pads each batch so all examples are the same length

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove the decoder start token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── Metric (Word Error Rate) ───────────────────────────────────────────────────

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids   = pred.predictions
    label_ids  = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str   = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str  = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer        = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = LANGUAGE
model.generation_config.task     = TASK
model.generation_config.forced_decoder_ids = None

# ── Training arguments ────────────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=500,
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=100,
    eval_steps=100,
    logging_steps=25,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    use_mps_device=False,
)

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")
