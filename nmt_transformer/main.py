import torch
from torch import nn
from utils import load_data, max_length, tokenizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import wandb
from huggingface_hub import login
import os


data = load_data()
huggingface_api = os.environ['HUGGINGFACE_API_KEY']
wandb_api = os.environ['WANDB_API_KEY']
login(token = huggingface_api)
wandb.login(key = wandb_api)

vocab_size = tokenizer.vocab_size
embedding_dim = 384
max_seq_len = max_length
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_heads = 8
num_layers = 4
batch_size = 16
hidden_size = 512
epochs = 5
#pytorch is cool

wandb.init(project = "Neural Machine Translation Transformer", entity = 'jenrola2292', name = 'first run')
wandb.config = {
   'learning_rate':2e-5, "epochs": epochs, "batch_size":batch_size
}


class EmbeddingEncoding(nn.Module): #the idea is to swap out this encoding layer with an lready pretrinad embedding model 
    def __init__(self, vocab_size = vocab_size, embedding_dim = embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    

class LearnedPositionalEncoding(nn.Module): #replacing fixed positional encoding with learnt positional encoding
    def __init__(self, max_seq_len = max_seq_len, embedding_dim = embedding_dim):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, input_embeddings): #this added embedding of either languages with positional embedding
        seq_len = input_embeddings.size(1)
        positions = torch.arange(seq_len, device = device).unsqueeze(0)
        positional_embeddings = self.positional_encoding(positions)

        input_embeddings = input_embeddings + positional_embeddings
        return input_embeddings
    
class CrossAttention(nn.Module):
    def __init__(self, num_heads = num_heads):
        #value and key come from the the encoder ie context
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids, context): #input_ids from Decoder, context from Encoder
        attention_output,_ = self.mha(query = input_ids, value = context, key = context)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    

class SelfAttention(CrossAttention):    
    def forward(self, input_ids):
        attention_output, _ = self.mha(query = input_ids, value = input_ids, key = input_ids)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    


class CausalSelfAttention(CrossAttention):
    def forward(self, input_ids, is_causal = True):
        attention_output, _ = self.mha(query = input_ids, value = input_ids, key = input_ids, is_causal = is_causal, attn_mask = self.generate_attn_mask(input_ids).to(device))
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    def generate_attn_mask(self, query_key_value, num_heads = num_heads):
        seq_len = query_key_value.size(1)   
        mask = torch.zeros(seq_len, seq_len)
        mask = mask.repeat(num_heads, 1, 1)  
        mask = mask.unsqueeze(0) 
        mask = mask.repeat(query_key_value.size(0), 1, 1, 1) 
        return mask.reshape(batch_size * num_heads, seq_len, seq_len)

    
#feed forward network
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim = embedding_dim, hidden_size = hidden_size, dropout_rate = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(embedding_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.add = torch.add
        self.layernorm = nn.LayerNorm(embedding_dim)
    def forward(self, input_ids):
        x = self.linear_1(input_ids)
        x = self.linear_2(x)
        x = self.dropout(x)
        x = self.add(input_ids, x) #residual connection
        x = self.layernorm(x)
        return x
    
#encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.ffn = FeedForward()
    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, dropout_rate = 0.1):
        super().__init__()
        self.embedder = EmbeddingEncoding()
        self.pos_embedding = LearnedPositionalEncoding()
        self.enc_layers = nn.ModuleList([
            EncoderLayer() for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.embedder(x)
        x = self.pos_embedding(x)
        for layer in self.enc_layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention()
        self.cross_attention = CrossAttention()
        self.ffn = FeedForward()
    
    def forward(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)
        x = self.ffn(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = EmbeddingEncoding()
        self.pos_embedding = LearnedPositionalEncoding()
        self.dec_layers = nn.ModuleList([
            DecoderLayer() for i in range(num_layers)
        ])
        self.ffn = FeedForward()
    def forward(self, x, context):
        x = self.embedder(x)
        x = self.pos_embedding(x)
        for layer in self.dec_layers:
            x = layer(x, context)
        x = self.ffn(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_layer = nn.Linear(embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,inputs):
        context, x = inputs  #ontext is the source language, x is target language
        context = self.encoder(context)
        x = self.decoder(x, context)
        x = self.final_layer(x)
        x = self.softmax(x) #brings out probabilities for each class
        return x


model = Transformer()
model.to(device)
optimizer = AdamW(model.parameters(), lr = 2e-5)
train_data, valid_data, test_data = load_data()

train_loader = DataLoader(train_data, batch_size = batch_size)
valid_loader = DataLoader(valid_data, batch_size = batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)
loss_func = nn.CrossEntropyLoss()

for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        en_input_ids = batch['en_input_ids'].to(device)
        es_input_ids = batch['es_input_ids'].to(device)
        es_target_ids = batch['es_target_ids'].to(device)
        output = model((en_input_ids, es_input_ids))
        loss = loss_func(output.view(-1, tokenizer.vocab_size),es_target_ids.view(-1))
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        wandb.log({'epoch': epoch+1,'training_loss': loss})


    training_loss /= len(train_loader)
    model.eval()

    for batch in valid_loader:
        en_input_ids = batch['en_input_ids'].to(device)
        es_input_ids = batch['es_input_ids'].to(device)
        es_target_ids = batch['es_target_ids'].to(device)
        output = model((en_input_ids, es_input_ids))
        loss = loss_func(output.view(-1, tokenizer.vocab_size),es_target_ids.view(-1))
        valid_loss += loss.item()
        wandb.log({'epoch': epoch+1,'validation_loss': loss})
    valid_loss /= len(valid_loader)

    print(f"Epoch {epoch + 1} : Train Loss : {training_loss} Valid Loss: {valid_loss}")

    torch.save(model.state_dict(), './model.pt')

        






