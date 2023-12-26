import datasets
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


def load_data():
    snli = datasets.load_dataset("snli", split="train")
    mnli = datasets.load_dataset("glue", "mnli", split="train")
    mnli = mnli.remove_columns(["idx"])
    snli = snli.cast(mnli.features)

    dataset = datasets.concatenate_datasets([snli, mnli])
    del snli, mnli
    dataset = dataset.filter(lambda x: True if x["label"] == 0 else False)

    dataset = dataset.map(
        lambda x: tokenizer(
            x["premise"], max_length=128, padding="max_length", truncation=True
        ),
        batched=True,
    )

    dataset = dataset.rename_column("input_ids", "anchor_ids")
    dataset = dataset.rename_column("attention_mask", "anchor_mask")

    dataset = dataset.map(
        lambda x: tokenizer(
            x["hypothesis"], max_length=128, padding="max_length", truncation=True
        ),
        batched=True,
    )

    dataset = dataset.rename_column("input_ids", "positive_ids")
    dataset = dataset.rename_column("attention_mask", "positive_mask")

    dataset = dataset.remove_columns(
        ["premise", "hypothesis", "label", "token_type_ids"]
    )
    dataset.set_format(type="torch", output_all_columns=True)
    return dataset
