import torch
from torch import nn
from utils import load_data, max_length, tokenizer


vocab_size = tokenizer.vocab_size
embedding_dim = 384
max_seq_len = max_length
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_heads = 6
batch_size = 16
class EmbeddingEncoding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embedding(x)
    

class LearnedPositionalEncoding(nn.Module): #replacing fixed positional encoding with learnt positional encoding
    def __init__(self, max_seq_len, embedding_dim):
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
        
    def forward(self, input_ids, context): #input_ids from query, context from key and values
        attention_output, attention_scores = self.mha(query = input_ids, value = context, key = context)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    

class SelfAttention(CrossAttention):    
    def forward(self, input_ids):
        attention_output, attention_scores = self.mha(query = input_ids, value = input_ids, key = input_ids)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    
def generate_attn_mask(query_key_value, num_heads = num_heads):
    seq_len = query_key_value.size(1)   
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.repeat(num_heads, 1, 1)  
    mask = mask.unsqueeze(0) 
    mask = mask.repeat(query_key_value.size(0), 1, 1, 1) 
    return mask.reshape(batch_size * num_heads, seq_len, seq_len)

class CausalSelfAttention(CrossAttention):
    def forward(self, input_ids, is_causal = True):
        attention_output, attention_scores = self.mha(query = input_ids, value = input_ids, key = input_ids, is_causal = is_causal, attn_mask = generate_attn_mask(input_ids))
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    
#feed forward network
    
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_size = 2048, dropout_rate = 0.1):
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




