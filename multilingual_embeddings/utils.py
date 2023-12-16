import datasets



def load_data():
    snli = datasets.load_dataset('snli', split='train')
    mnli = datasets.load_dataset('glue', 'mnli', split='train')
    mnli = mnli.remove_columns(['idx'])
    snli = snli.cast(mnli.features)

    dataset = datasets.concatenate_datasets([snli, mnli])
    del snli, mnli
    dataset = dataset.filter(
        lambda x: True iff x['label'] == 0 else False
    )
    return dataset


