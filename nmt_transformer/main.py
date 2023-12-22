import torch
from torch import nn
from utils import load_data, max_length, tokenizer


vocab_size = tokenizer.vocab_size
embedding_dim = 384
max_seq_len = max_length
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_heads = 6
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
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(embedding_dim)
        
    def forward(self, input_ids, context): #input_ids from query, context from key and values
        attention_output, attention_scores = self.mha(query = input_ids, value = context, key = context)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    

class SelfAttention(nn.Module):
    def __init__(self, num_heads = num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.layernorm = nn.LayerNorm(embedding_dim)
    
    def forward(self, input_ids):
        attention_output, attention_scores = self.mha(query = input_ids, value = input_ids, key = input_ids)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    
class CausalSelfAttention(CrossAttention):
    def forward(self, input_ids, attention_mask, is_causal = True):
        attention_output, attention_scores = self.mha(query = input_ids, value = input_ids, key = input_ids, is_causal = is_causal, attn_mask = attention_mask)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x
    
#feed forward network

