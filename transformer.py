import torch
from torch import nn
import math
from configuration import TransformerConfig
import torch.optim as optim


# Simple linear layer to change dimension of input features to d_model
class InputEmbedding(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(self.config.d_in)
        self.linear = nn.Linear(self.config.d_in, self.config.d_model)

    def forward(self, x):

        # Normalize input features
        x = self.norm(x)

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
        pe = torch.zeros(config.seq_len, config.d_model)

        # Compute sin on even columns only
        pe[:, 0::2] = torch.sin(self.angles)

        # Compute cos on odd columns only
        pe[:, 1::2] = torch.cos(self.angles)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        B, S, D = x.shape

        x = x + self.pe[:S].to(x.device)
        return x


class Attention(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()

        self.config = config

        # ensure d_model is divisible by n_heads
        assert self.config.d_model % self.config.n_heads == 0
        
        self.norm = nn.LayerNorm(self.config.d_model)
        self.w_q = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.w_k = nn.Linear(self.config.d_model, self.config.d_model, bias=False)
        self.w_v = nn.Linear(self.config.d_model, self.config.d_model, bias=False)

        # Output projection layer
        self.w_o = nn.Linear(self.config.d_model, self.config.d_model)

    def forward(self, x):
        
        residual = x
        x = self.norm(x)

        # B: batch (number of examples per batch)
        # S: sequence length (number of tokens or timesteps)
        # D: model dimension (embedding size per token)
        # H: number of attention heads (heads in multi-head attention)
        # D // H: per-head dimension (dimension per head = d_model / num_heads)

        B, S, D = x.size()
        H = self.config.n_heads

        # Multiply input by q/k/v weights and reshape the input into multiple heads
        # Transpose so that head dim comes before sequence dim (computation convenience)
        q = self.w_q(x).view(B, S, H, D // H).transpose(1,2) # Queries
        k = self.w_k(x).view(B, S, H, D // H).transpose(1,2) # Keys
        v = self.w_v(x).view(B, S, H, D // H).transpose(1,2) # Values

        # sqrt is done to prevent large values from dominating the normalization
        atn_matrix = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(D // H) # b h s s
        atn_weights = torch.softmax(atn_matrix, dim=-1)
        out = torch.matmul(atn_weights, v) # b h s d//h

        # Stacks the heads back together continuously in memory
        atn_output = out.transpose(1,2).contiguous().view(B, S, D)
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
        self.policy_head = nn.Linear(config.d_model, config.d_out_policy, bias=False)
        self.value_head = nn.Linear(config.d_model, config.d_out_value, bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.scheduler = None
        self.scaler = None

    def forward(self, x):
        x = self.embeddinng(x)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Pooling: take mean of the sequence dimension to compress it
        # x = x.mean(dim=1)

        # Last token pooling
        x = x[:, -1, :]

        policy_out = self.policy_head(x)
        value_out = self.value_head(x)
        
        return torch.distributions.Categorical(logits=policy_out), value_out
    
    def optim_step(self):
        self.optimizer.step()

    def optim_zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self, *args, **kwargs):

        return {
            "model": super().state_dict(*args, **kwargs),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if getattr(self, "scheduler", None) else None,
            "scaler": self.scaler.state_dict() if getattr(self, "scaler", None) else None
        }

    def load_state_dict(self, state, *args, **kwargs):
        super().load_state_dict(state["model"], *args, **kwargs)
        self.optimizer.load_state_dict(state["optimizer"])
        if getattr(self, "scheduler", None) and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if getattr(self, "scaler", None) and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
