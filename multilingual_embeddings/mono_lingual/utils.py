from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


def load_data():
    dataset_url = "odunola/yoruba-english-pairs"
    dataset = load_dataset(dataset_url, split="train")

    dataset = dataset.map(
        lambda x: tokenizer(
            x["eng"], max_length=128, padding="max_length", truncation=True
        ),
        batched=True,
    )
    dataset = dataset.rename_column("input_ids", "english_ids")
    dataset = dataset.rename_column("attention_mask", "english_mask")
    dataset = dataset.rename_column("token_type_ids", "english_token_ids")
    dataset = dataset.map(
        lambda x: tokenizer(
            x["yor"], max_length=256, padding="max_length", truncation=True
        ),
        batched=True,
    )
    dataset = dataset.rename_column("input_ids", "mono_ids")
    dataset = dataset.rename_column("attention_mask", "mono_mask")
    dataset = dataset.rename_column("token_type_ids", "mono_token_ids")
    dataset = dataset.remove_columns(["eng", "yor"])
    dataset.set_format(type="torch", output_all_columns=True)
    return dataset
