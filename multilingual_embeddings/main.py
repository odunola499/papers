from utils import load_data
import os
from torch import nn
from transformers import AutoTokenizer, AutoModel


huggingface_key = os.environ['HUGGINGFACE_API_KEY']

dataset = load_data()


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
                                  
class Pooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 768)
        self.a = nn.Linear(768, 768, bias = True)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.layer(x)
        x = self.a(x)
        x = self.activation(x)
        return x
    

pooler = Pooler()
model.pooler = Pooler()
#now we have our model

#we need to freeze all layers except the pooling module

for name, param in model.named_parameters():
    if 'pooler' not in name:
        param.__requires_grad = False  #freezing layes

#now we need to finetune the model

dataset = dataset.map(
    lambda x: tokenizer(
        x['premise'], max_length = 128, padding = 'max_length',
        truncation = True
    ), batched = True
)

dataset = dataset.rename_column('input_ids', 'anchor_ids')
dataset = dataset.rename_column('attention_mask', 'anchor_mask')

dataset = dataset.map(
    lambda x: tokenizer(
            x['hypothesis'], max_length=128, padding='max_length',
            truncation=True
    ), batched=True
)

dataset = dataset.rename_column('input_ids', 'positive_ids')
dataset = dataset.rename_column('attention_mask', 'positive_mask')

dataset = dataset.remove_columns(['premise', 'hypothesis', 'label', 'token_type_ids'])