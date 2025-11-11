import torch
from torch import nn
import math
from hyperparams import TransformerConfig


# Simple linear layer to change dimension of input features to d_model
class InputEmbedding(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(self.config.d_in, self.config.d_model)
        
    def forward(self, x):

        # Multiply by sqrt(d_model) to keep post-embedding variance stable
        return self.linear(x) * math.sqrt(self.config.d_model)
    

# Cosine positional encoding to give each input position a unique representation
class CosineEncode(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        
        # Generate vector that expresses position, along seq_len dim
        self.pos = torch.arange(config.seq_len).unsqueeze(1)

        # Generate vector for sin/cos computations along d_model dim
        self.i = torch.arange(0, config.d_model, 2)

        # Compute angles for each position, seq_len*d_model/2
        self.angles = self.pos / (10000 ** (self.i / config.d_model))

        # Initialize positional encoding matrix
        self.pe = torch.zeros(config.seq_len, config.d_model)

        # Compute sin on even columns only
        self.pe[:, 0::2] = torch.sin(self.angles)

        # Compute cos on odd columns only
        self.pe[:, 1::2] = torch.cos(self.angles)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe.to(x.device)
        return x

class Attention(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()

        # ensure d_model is divisible by n_heads
        assert config.d_model % config.n_heads == 0
        
        self.norm = nn.LayerNorm(config.d_model)
        self.w_q = nn.Parameter(torch.empty(config.n_heads, config.d_model, config.d_model // config.n_heads))
        self.w_k = nn.Parameter(torch.empty(config.n_heads, config.d_model, config.d_model // config.n_heads))
        self.w_v = nn.Parameter(torch.empty(config.n_heads, config.d_model, config.d_model // config.n_heads))
        
        # Output projection layer
        self.w_o = nn.Linear(config.d_model, config.d_model)

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)


    def forward(self, x):
        
        residual = x
        x = self.norm(x)

        # b	batch (number of examples per batch)
        # s = k = v, sequence length (number of tokens or timesteps)
        # i	input dimension	(embedding size per token)
        # h	number of attention heads (heads in multi-head attention)
        # d	per-head dimension (dimension per head = total_dim / num_heads)

        q = torch.einsum('bsi,hid->bhsd', x, self.w_q) # Queries
        k = torch.einsum('bsi,hid->bhsd', x, self.w_k) # Keys
        v = torch.einsum('bsi,hid->bhsd', x, self.w_v) # Values

        # sqrt is done to prevent large values from dominating the normalization
        atn_matrix = torch.einsum('bhsd,bhkd->bhsk', q, k) / math.sqrt(k.size(-1))
        atn_weights = torch.softmax(atn_matrix, dim=-1)
        out = torch.einsum('bhsk,bhkd->bhsd',atn_weights, v)

        # Important line - stacks the heads back together continuously in memory
        atn_output = out.transpose(1,2).contiguous().view(x.size(0), x.size(1), -1)
        atn_output = self.w_o(atn_output)

        # add atn_output to the original input (residual connection)
        return atn_output + residual

class MLP(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.layer1 = nn.Linear(config.d_model, config.mlp_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(config.mlp_dim, config.d_model)
        # Omit dropout for now - can cause instability with PPO
        # self.dropout = nn.Dropout(config.dropout_rate)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.embeddinng = InputEmbedding(config)
        self.pos_encoding = CosineEncode(config)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.policy_head = nn.Linear(config.d_model, config.d_out_policy)
        self.value_head = nn.Linear(config.d_model, config.d_out_value)

    def forward(self, x):
        x = self.embeddinng(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Pooling: take mean of the sequence dimension to compress it
        x = x.mean(dim=1)

        policy_out = self.policy_head(x)
        value_out = self.value_head(x)
        return policy_out, value_out
