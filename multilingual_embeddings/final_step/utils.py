from datasets import load_dataset
from transformers import AutoTokenizer
import datasets
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')

def remove_nonetypes(pairs):
    eng = pairs['eng']
    yor = pairs['yor']
    good_eng, good_yor = [], []
    for i , j in zip(eng, yor):
        if (len(i) < 4 or len(j) < 4) or (i == None or j == None):
            continue
        good_eng.append(i)
        good_yor.append(j)
    return {'eng': good_eng, 'yor':good_yor}

def load_data_pairs():
    dataset_url = 'odunola/yoruba-english-pairs'
    dataset = load_dataset(dataset_url, split = 'train')
    dataset = dataset.map(lambda x: remove_nonetypes(x), batched = True).shuffle().select(range(150000))
    dataset = dataset.map(
        lambda x: tokenizer(
            x['eng'], max_length = 256, padding = 'max_length', truncation = True
        ), batched = True
    )
    dataset = dataset.rename_column('input_ids', 'positive_ids')
    dataset = dataset.rename_column('attention_mask', 'positive_mask')
    dataset = dataset.rename_column('token_type_ids', 'positive_token_ids')
    dataset = dataset.map(
        lambda x: tokenizer(
            x['yor'], max_length = 256, padding = 'max_length', truncation = True
        ), batched = True
    )
    dataset = dataset.rename_column('input_ids', 'anchor_ids')
    dataset = dataset.rename_column('attention_mask', 'anchor__mask')
    dataset = dataset.rename_column('token_type_ids', 'anchor_token_ids')
    dataset = dataset.remove_columns(['eng', 'yor', ])
    #dataset.set_format(type = 'torch', output_all_columns=True)
    return dataset


def load_data_embs():
    snli = load_dataset('snli', split='train')
    mnli = load_dataset('glue', 'mnli', split='train')
    mnli = mnli.remove_columns(['idx'])
    snli = snli.cast(mnli.features)

    dataset = datasets.concatenate_datasets([snli, mnli])
    del snli, mnli
    dataset = dataset.filter(
        lambda x: True if x['label'] == 0 else False
    )
    dataset = dataset().shuffle().select(range(150000))
    
    dataset = dataset.map(
            lambda x: tokenizer(
                x['premise'], max_length = 256, padding = 'max_length',
                truncation = True
            ), batched = True
            )

    dataset = dataset.rename_column('input_ids', 'anchor_ids')
    dataset = dataset.rename_column('attention_mask', 'anchor_mask')
    dataset = dataset.rename_column('token_type_ids', 'anchor_token_ids')
    dataset = dataset.map(
        lambda x: tokenizer(
                x['hypothesis'], max_length=256, padding='max_length',
                truncation=True
        ), batched=True
    )

    dataset = dataset.rename_column('input_ids', 'positive_ids')
    dataset = dataset.rename_column('attention_mask', 'positive_mask')
    dataset = dataset.rename_column('token_type_ids', 'positive_token_ids')

    dataset = dataset.remove_columns(['premise', 'hypothesis', 'label'])
    #dataset.set_format(type='torch', output_all_columns=True)
    return dataset


def load_data():
    data_1, data_2 = load_data_pairs(), load_data_embs()
    dataset = datasets.concatenate_datasets([data_1, data_2])
    dataset.set_format(type='torch', output_all_columns=True)
    return dataset.shuffle()
