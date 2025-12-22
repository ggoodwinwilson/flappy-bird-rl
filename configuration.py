from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MLPConfig:
    d_in: int = 180
    d_out_actor: int = 2
    d_out_critic: int = 1
    d_model: int = 256
    d_model_critic: int = 256
    learning_rate: float = 3e-4
    num_epochs: int = 3
    batch_size: int = 32
    dtype: torch.dtype = torch.float32
    seq_len: int = 10 # Only used for transformers, kept for compatibility
    model_type = "mlp"

    def as_dict(self):
        return self.__dict__

mlp_config = MLPConfig()


@dataclass(frozen=True)
class TransformerConfig:
    model_type = "xfmr"
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 4
    mlp_dim: int = d_model * 4
    dropout_rate: float = 0.1
    seq_len: int = 10
    d_in: int = 180
    d_out_policy: int = 2
    d_out_value: int = 1
    num_epochs: int = 3
    batch_size: int = 32
    dtype: torch.dtype = torch.float32
    learning_rate: float = 3e-4

    def as_dict(self):
        return self.__dict__

transformer_config = TransformerConfig()


@dataclass(frozen=True)
class PPOConfig:
    rl_type: str = 'ppo'
    td_lambda: float = 0.95
    gamma: float = 0.99
    eps_clip: float = 0.2
    ent_coef: float = 0.02
    critic_coef: float = 0.5
    learning_rate: float = 3e-4
    rollout_len: int = 256
    total_train_steps: int = 100_000
    d_in: int = 180

    def as_dict(self):
        return self.__dict__
    
ppo_config = PPOConfig()

def make_hparams_dict(rl_config, model_config):

    d = {}

    for k, v in rl_config.as_dict().items():
        d[f"rl/{k}"] = v

    for k, v in model_config.as_dict().items():
        d[f"model/{k}"] = v

    # Make sure dict is compatible with Tensorboard
    for k in d.keys():
        if isinstance(d[k], torch.dtype):
            d[k] = str(d[k])

    return d
