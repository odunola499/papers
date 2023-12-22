from utils import load_data, tokenizer
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
#first we will use a new tokenizer

vocab_size = tokenizer.vocab_size
embedding_dim = 384
units = 128
train_data, valid_data, test_data = load_data(subset = True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
target_embedder = nn.Embedding(tokenizer.vocab_size,units)

class Encoder(nn.Module):
    def __init__(self, embedding_dim = embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, units, bidirectional = True, batch_first = True)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x,_ = self.rnn(x)
        x = x[:,:,units:] + x[:,:,:units]
        return x

#value is context, 
#attention
class CrossAttention(nn.Module):
    def __init__(self, units = units):
        super().__init__()
        self.mha = nn.MultiheadAttention(units, num_heads=1, batch_first = True)
        self.layernorm = nn.LayerNorm(units)
    def forward(self, input_ids, context):
        attention_output, attention_scores = self.mha(query = input_ids, value = context, key = context)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x

#decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim,units)
        self.attention = CrossAttention(units)
        self.fc = nn.Linear(units, vocab_size)
    def forward(self, context, targ_in, state = None):
        x = self.embedding(targ_in)

        x, state = self.rnn(x, state)

        x = self.attention(x, context)
        logits = self.fc(x)

        return logits


class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        #now we write training loop
    
    def forward(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits
    

model = Translator()
model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total trainable parameters: {total_params}")


optimizer = AdamW(model.parameters())
loss_func = CrossEntropyLoss()
epochs = 20
#batch_size = input('input batch size: ')
batch_size = 32
train_loader = DataLoader(train_data, batch_size = batch_size)
valid_loader = DataLoader(valid_data, batch_size = batch_size)
test_loader = DataLoader(test_data, batch_size = batch_size)


print(f"Total trainable parameters: {total_params}")
for epoch in range(epochs):
    train_loss = 0.0
    model.train()
    for batch in tqdm(train_loader):
        context_ids = batch['en_input_ids'].to(device)
        target_in_ids = batch['es_input_ids'].to(device)
        target_out_ids = batch['es_target_ids'].to(device)
        optimizer.zero_grad()
        output = model(context_ids, target_in_ids)
        loss = loss_func(output.view(-1, tokenizer.vocab_size),target_out_ids.view(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)

    eval_loss = 0.0
    model.eval()
    for batch in tqdm(valid_loader):
        context_ids = batch['en_input_ids'].to(device)
        target_in_ids = batch['es_input_ids'].to(device)
        target_out_ids = batch['es_target_ids'].to(device)
        with torch.no_grad():
            output = model(context_ids, target_in_ids)
            loss = loss_func(output.view(-1, tokenizer.vocab_size),target_out_ids.view(-1))
        eval_loss += loss.item()
    eval_loss /= len(valid_loader)
    torch.save(model.state_dict(), 'model.pth')

    print(f"after epoch {epoch + 1}: Training Loss is {train_loss}")
    print(f"after epoch {epoch + 1}: Training Loss is {eval_loss}")

    
