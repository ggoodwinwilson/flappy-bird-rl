import torch
from torch import nn
import math
from hyperparams import TransformerConfig


# Simple linear layer to change dimension of input features to d_model
class InputEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_in, d_model)
        
    def forward(self, x):

        # Multiply by sqrt(d_model) to keep post-embedding variance stable
        return self.linear(x) * math.sqrt(self.d_model)
    

# Cosine positional encoding to give each input position a unique representation
class CosineEncode(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        
        # Generate vector that expresses position, along seq_len dim
        self.pos = torch.arange(seq_len).unsqueeze(1)

        # Generate vector for sin/cos computations along d_model dim
        self.i = torch.arange(0, d_model, 2)

        # Compute angles for each position, seq_len*d_model/2
        self.angles = self.pos / (10000 ** (self.i / d_model))

        # Initialize positional encoding matrix
        self.pe = torch.zeros(seq_len, d_model)

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

class MLP(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.layer1 = nn.Linear(config.d_model, config.mlp_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(config.mlp_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x):
        
