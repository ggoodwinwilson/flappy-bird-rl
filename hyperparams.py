
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class PPOConfig:
    td_lambda: float = 0.95
    gamma: float = 0.99
    eps_clip: float = 0.2
    ent_coef: float = 0.02
    critic_coef: float = 0.5
    learning_rate: float = 3e-4
    num_epochs: int = 3
    batch_size: int = 32
    rollout_len: int = 128
    total_timesteps: int = 10_000_000
    dtype: torch.dtype = torch.float32
    d_in: int = 180
    d_out_actor: int = 2
    d_out_critic: int = 1
    d_hidden_actor: int = 512
    d_hidden_critic: int = 512

    def as_dict(self):
        return self.__dict__
    
config = PPOConfig()