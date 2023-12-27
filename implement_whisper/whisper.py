import torch
from torch import nn
import torch.nn.functional as F


in_channels = 80
out_channels = 80
kernel_size = 3
padding = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
num_heads = 6  # number of heads of original whisper tiny
hidden_size = 384
num_layers = 4
batch_size = 8
encoder_embed_dim = 1500
decoder_embed_dim = 1500
decoder_max_len = 512


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=decoder_max_len, embedding_dim=decoder_embed_dim):
        super().__init__()
        self.positional_encoding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(
        self, input_embeddings
    ):  # this added embedding of either languages with positional embedding
        seq_len = input_embeddings.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positional_embeddings = self.positional_encoding(positions)

        input_embeddings = input_embeddings + positional_embeddings
        return input_embeddings


class SinusoidPositionalEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
    ):
        super(SinusoidPositionalEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))

        batch_size, channels, sequence_length = x.size()
        position = (
            torch.arange(sequence_length, dtype=torch.float).unsqueeze(1).to(x.device)
        )
        pos_embedding = torch.zeros(sequence_length, channels).to(device)
        pos_embedding[:, 0::2] = torch.sin(
            position
            * (1 / (10000 ** (2 * torch.arange(0, channels, 2).float() / channels))).to(
                x.device
            )
        )  # even
        pos_embedding[:, 1::2] = torch.cos(
            position
            * (1 / (10000 ** (2 * torch.arange(0, channels, 2).float() / channels))).to(
                x.device
            )
        )  # odd

        pos_embedding_pe = pos_embedding.unsqueeze(0)
        pos_embedding_pe = pos_embedding.permute(0, 2, 1)
        x = x + pos_embedding_pe

        return x


# output of sinousodial is [batch_size, channel_size, sequence_length /2]


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        attention_output, _ = self.mha(query=input_ids, value=input_ids, key=input_ids)
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_size=hidden_size, dropout_rate=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.add = torch.add
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        x = self.linear_1(input_ids)
        x = self.linear_2(x)
        x = self.dropout(x)
        x = self.add(input_ids, x)  # residual connection
        x = self.layernorm(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention(embed_dim=encoder_embed_dim)
        self.mlp = MLP(embed_dim=encoder_embed_dim)

    def forward(self, input_ids):
        x = self.attention(input_ids)
        x = self.mlp(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers=num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.incoming_layer = SinusoidPositionalEmbedding()
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(num_layers)])

    def forward(self, input_ids):
        input_ids = self.incoming_layer(input_ids)
        for layer in self.encoder_blocks:
            input_ids = layer(input_ids)
        return input_ids


# decoder
# decoder takes in
class CausalSelfAttention(SelfAttention):
    def forward(self, input_ids, is_causal=True):
        attention_output, _ = self.mha(
            query=input_ids,
            value=input_ids,
            key=input_ids,
            is_causal=is_causal,
            attn_mask=self.generate_attn_mask(input_ids).to(device),
        )
        x = torch.add(input_ids, attention_output)
        x = self.layernorm(x)
        return x

    def generate_attn_mask(self, query_key_value, num_heads=num_heads):
        seq_len = query_key_value.size(1)
        mask = torch.zeros(seq_len, seq_len)
        mask = mask.repeat(num_heads, 1, 1)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(query_key_value.size(0), 1, 1, 1)
        return mask.reshape(batch_size * num_heads, seq_len, seq_len)


class CrossAttention(SelfAttention):
    def forward(self, input_ids, context):
        attention_output, _ = self.mha(
            query=input_ids,
            key=context,
            value=context,
        )
        x = torch.add(input_ids, attention_output)
        return self.layernorm(x)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(embed_dim=decoder_embed_dim)
        self.cross_attention = CrossAttention(embed_dim=decoder_embed_dim)
        self.mlp = MLP(embed_dim=decoder_embed_dim)

    def forward(self, input_ids, context):
        out = self.causal_self_attention(input_ids)
        out = self.cross_attention(out, context)
        out = self.mlp(out)
        return out
