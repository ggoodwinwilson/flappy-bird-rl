import numpy as np
import torch
import random
from agent import Agent
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from configuration import PPOConfig

@dataclass
class Checkpoint:
    model: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Optional[Dict[str, Any]]
    scaler: Optional[Dict[str, Any]]
    epoch: int
    global_step: int
    batch_idx: int
    rng_state: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Checkpoint":
        return Checkpoint(**d)

class CheckpointManager:
    def __init__(self, agent:Agent, rl_config:PPOConfig):
        
        
        self.agent = agent
        self.rl_config = rl_config or {}
        self.model_config = agent.model_config
        self.epoch = 0
        self.global_step = 0
        self.batch_idx = 0

    def save(self, path):
        ckpt = Checkpoint(
            model=self.agent.model.state_dict(),
            optimizer=self.agent.model.optimizer.state_dict(),
            # scheduler=self.agent.scheduler.state_dict() if self.agent.scheduler else None,
            # scaler=self.agent.scaler.state_dict() if self.agent.scaler else None,
            epoch=self.epoch,
            global_step=self.global_step,
            batch_idx=self.batch_idx,
            # rng_state=self._collect_rng(),
            meta=self.config,
        )
        torch.save(ckpt.to_dict(), path)

    def load(self, path, map_location="cpu"):
        raw = torch.load(path, map_location=map_location)
        ckpt = Checkpoint.from_dict(raw)

        self.agent.model.load_state_dict(ckpt.model)
        self.agent.model.optimizer.load_state_dict(ckpt.optimizer)
        self.epoch = ckpt.epoch
        self.global_step = ckpt.global_step
        self.batch_idx = ckpt.batch_idx

        # if self.agent.scheduler and ckpt.scheduler:
        #     self.agent.scheduler.load_state_dict(ckpt.scheduler)
        # if self.agent.scaler and ckpt.scaler:
        #     self.agent.scaler.load_state_dict(ckpt.scaler)
        # self._restore_rng(ckpt.rng_state)

def make_run_id(agent: Agent, env_name, rl_type, tag=None):
    parts = [
        env_name,
        rl_type,
        agent.model_config.model_type,
        f"dim{agent.model_config.d_model}",
        tag,
    ]
    return "_".join(p for p in parts if p)

def make_paths(agent: Agent, env_name, rl_type, tag=None):
    run_id = make_run_id(agent, env_name, rl_type, tag)

    return {
        "run_id": run_id,
        "tensorboard_dir": f"runs/{run_id}",
        "checkpoint_recent": f"checkpoints/{run_id}_recent.pth",
        "checkpoint_best": f"checkpoints/{run_id}_best.pth",
    }


# def save_models(model:Agent, path, global_step, max_avg_rew):
#     torch.save({
#         "model": model.state_dict(),
#         "ppo_config": model.ppo_config.as_dict(),
#         "model_config": model.model_config.as_dict(),
#         "max_avg_rew": max_avg_rew,
#         "global_step": global_step
#     }, path)

# def load_models(model:Agent, path):
#     bundle = torch.load(path, map_location=model.device)
#     if bundle["ppo_config"] != model.ppo_config.as_dict() or \
#         bundle["model_config"] != model.model_config.as_dict():
#         raise ValueError("Checkpoint config mismatch")
#     model.load_state_dict(bundle["model"])
#     avg_rew = bundle.get("avg_rew", float('-inf'))
#     global_step = bundle.get("global_step", float('-inf'))
#     return avg_rew, global_step

