import torch
import gc
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Load data
train_df = pd.read_csv("TrainingOnlineData.csv",nrows = 7000 )
test_df = pd.read_csv("TestingOnlineData.csv")

# Rename columns
train_df.columns = ["audio", "sentence"]
test_df.columns = ["audio", "sentence"]

# Convert pandas dataframes to datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Convert the sample rate of every audio file
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load feature extractor, tokenizer, and processor
feature_extractor = WhisperFeatureExtractor.from_pretrained("quinnb/whisper-Large-v3-hindi")
tokenizer = WhisperTokenizer.from_pretrained("quinnb/whisper-Large-v3-hindi", language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained("quinnb/whisper-Large-v3-hindi", language="Hindi", task="transcribe")


def prepare_dataset(examples):
    audio = examples["audio"]
    examples["input_features"] = feature_extractor(audio["array"], sampling_rate=16000).input_features[0]
    del examples["audio"]
    sentences = examples["sentence"]
    examples["labels"] = tokenizer(sentences).input_ids
    del examples["sentence"]
    return examples


# Map the prepare_dataset function
train_dataset = train_dataset.map(prepare_dataset, num_proc=1)
test_dataset = test_dataset.map(prepare_dataset, num_proc=1)

# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Load pre-trained model
model = WhisperForConditionalGeneration.from_pretrained("quinnb/whisper-Large-v3-hindi")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Training arguments
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-Large-v3-hindiCustomData",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=False,
    generation_max_length=225,
    save_steps=10000, #increase save and eval to like 10k
    eval_steps=10000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
model.to(device)
# Trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Free up cache before training
torch.cuda.empty_cache()
gc.collect()
processor.save_pretrained(training_args.output_dir)
# Start training
trainer.train()

# Push to hub
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_17_0",
    "dataset": "Custom Hindi dataset",
    "dataset_args": "config: hi, split: test",
    "language": "hi",
    "model_name": "Whisper Large v3 Trained on Hindi",
    "finetuned_from": "quinnb/whisper-Large-v3-hindi",
    "tasks": "automatic-speech-recognition",
}
trainer.push_to_hub(**kwargs)

# Free up cache after training
torch.cuda.empty_cache()
gc.collect()
