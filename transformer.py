import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        
    def forward(self, x):
        return self.linear(x)
    
    
class CosineEncode(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        
        # Generate vector that expresses position, along seq_len dim
        pos = torch.arange(seq_len).unsqueeze(1)

        # Generate vector for sin/cos computations along d_model dim
        i = torch.arange(0, d_model, 2)

        # Compute angles for each position, seq_len*d_model/2
        angles = pos / (10000 ** (i / d_model))

        # Initialize positional encoding matrix
        pe = torch.zeros(seq_len, d_model)

        # Compute sin on even columns only
        pe[:, 0::2] = torch.sin(angles)

        # Compute cos on odd columns only
        pe[:, 1::2] = torch.cos(angles)