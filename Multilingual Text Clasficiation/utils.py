from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

model_url = "bert-base-multilingual-uncased"
data_url = "odunola/multilingual-sentiments"
tokenizer = AutoTokenizer.from_pretrained(model_url)
max_length = 256
softmax = torch.nn.Softmax(dim=-1)


def process_batch(batch):
    text = batch["text"]
    label = batch["label"]
    text_tensors = tokenizer(text, max_length=max_length, truncation=True, padding=True)
    input_ids = text_tensors["input_ids"]
    attention_mask = text_tensors["attention_mask"]
    label_tensors = torch.tensor(label)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": label_tensors,
    }


def load_data():
    data = load_dataset(data_url)["train"]
    data = data.map(lambda batch: process_batch(batch), batched=True)
    data = data.remove_columns(["text"])
    data.set_format(type="torch", output_all_columns=True)
    train_valid = data.train_test_split(test_size=0.3, shuffle=True)
    train_data = train_valid["train"]
    valid = train_valid["test"]
    valid = valid.train_test_split(test_size=0.2)
    test_data = valid["test"]
    valid_data = valid["train"]
    return train_data, valid_data, test_data


#
def calculate_accuracy(logits, target):
    soft = softmax(logits)
    return np.mean((torch.argmax(soft, dim=-1) == target).numpy())
