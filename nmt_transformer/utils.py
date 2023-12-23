#code here retrieves preprocessed data
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
#english to spanish
max_length = 512
def get_es_input_target(batch, max_length=max_length):
    # Tokenize the Spanish translation with no padding/truncation
    tokens = tokenizer([i['es'] for i in batch['translation']], add_special_tokens=False)
    ids = [i[:max_length] for i in tokens['input_ids']] #for truncation

    new_ids_target, new_masks_target, new_token_ids_target = [], [], []
    for id in ids:
        new_id = id
        length = len(new_id)
        extra_length = max_length - length
        new_id = new_id + [tokenizer.pad_token_id]* extra_length
        assert len(new_id) == max_length
        new_ids_target.append(new_id)
        new_mask = [1]* length + [0]* extra_length
        new_token_id = [0] * max_length
        new_masks_target.append(new_mask)
        new_token_ids_target.append(new_token_id)
        
    new_ids_input, new_masks_input, new_token_ids_input = [], [], [] 
    for id in ids:
        new_id = id[:-1]
        new_id = [tokenizer.bos_token_id] + new_id
        length = len(new_id)
        extra_length = max_length - length
        new_id = new_id + [tokenizer.pad_token_id]* extra_length
        assert len(new_id) == max_length
        new_ids_input.append(new_id)
        new_mask = [1]* length + [0]* extra_length
        new_token_id = [0] * max_length
        new_masks_input.append(new_mask)
        new_token_ids_input.append(new_token_id)
    return {
        'es_input_ids' : new_ids_input,
        'es_mask_input' : new_masks_input,
        'es_token_ids_input': new_token_ids_input,
        'es_target_ids': new_ids_target,
        'es_masks_target': new_masks_target,
        'es_token_ids_target': new_token_ids_target
    }


def preprocess_target_out(batch):
    texts = [i['es'] for i in batch['translation']]

def preprocess_data(dataset, max_length = max_length):
    dataset = dataset.map(
        lambda x: tokenizer([i['en'] for i in x['translation']], max_length = max_length, padding = 'max_length', truncation = True), batched = True
    )
    dataset = dataset.rename_column('input_ids', 'en_input_ids')
    dataset = dataset.rename_column('attention_mask', 'en_attention_mask')
    dataset = dataset.rename_column('token_type_ids', 'en_token_type_ids')

    dataset = dataset.map(
       lambda x: get_es_input_target(x), batched = True
    )

    dataset = dataset.remove_columns(['translation'])
    dataset.set_format(type = 'torch', output_all_columns = True)
    return dataset

def load_data(subset = True):
    if subset:
        dataset = load_dataset('opus100','en-es')['validation']
    else:
        dataset = load_dataset('opus100','en-es')['train']
    #next we split the data
    splits = dataset.train_test_split(test_size = 0.3)
    train_data = splits['train']
    valid_test_data = splits['test'].train_test_split(test_size = 0.2)
    valid_data = valid_test_data['train']
    test_data = valid_test_data['test']
    train_data = preprocess_data(train_data)
    valid_data = preprocess_data(valid_data)
    test_data = preprocess_data(test_data)
    return train_data, valid_data, test_data