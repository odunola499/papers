from datasets import load_dataset
from huggingface_hub import login
import os
from transformers import WhisperProcessor
import torch
from typing import Any, Dict, List, Union


def process_data(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    sentences = batch['sentence']
    input_features = feature_extractor(audio["array"], sampling_rate = audio['sampling_rate']).input_features[0]
    labels = tokenizer(sentences).input_ids
    return {
        "input_features":input_features,
        "labels": labels
    }

def load_data():
    data = load_dataset(data_url)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    data = data.map(lambda batch: process_data(batch, feature_extractor, tokenizer))
    data = data.train_test_split(test_size = 0.2)
    train_data = data['train']
    valid_data = data['test']
    return train_data, valid_data

class DataCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    

login(os.environ['HUGGINGFACE_API_KEY'])
model_url = 'openai/whisper-small'
data_url = 'odunola/yoruba_audio_data'
processor = WhisperProcessor.from_pretrained(model_url, language = 'Yoruba')
data_collator = DataCollator(processor = processor)