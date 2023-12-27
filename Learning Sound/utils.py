from datasets import load_dataset
from transformers import WhisperProcessor
import torch
from typing import Any, Dict, List, Union
import evaluate


def process_data(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    sentences = batch["sentence"]
    input_features = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    labels = tokenizer(sentences).input_ids
    return {"input_features": input_features, "labels": labels}


def load_data():
    data = load_dataset(data_url)["train"].shuffle()
    data = data.train_test_split()
    train_data = data["train"]
    valid_data = data["test"]
    return train_data, valid_data


class DataCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt", padding="max_length", max_length=256
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


model_url = "openai/whisper-small"
data_url = "odunola/yoruba-audio-preprocessed-2"
processor = WhisperProcessor.from_pretrained(model_url, language="Yoruba", task = "transcribe")
data_collator = DataCollator(processor=processor)
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
