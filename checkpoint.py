import numpy as np
import torch
import random
from agent import Agent
import dataclasses
from typing import Optional, Dict, Any
from configuration import PPOConfig
import os

class CheckpointManager:
    def __init__(self, agent: Agent, rl_config: PPOConfig):
        self.agent = agent
        self.rl_config = rl_config
        self.model_config = agent.model_config

        # training state you want to track
        self.epoch = 0
        self.training_step = 0
        self.batch_idx = 0
        self.max_high_score = 0
        self.max_1k_rew = float("-inf")

    # Helpers for scalar training state
    def _train_keys(self):
        # Add new scalar fields here only
        return [
            "epoch",
            "training_step",
            "batch_idx",
            "max_high_score",
            "max_1k_rew",
        ]

    def _get_train_state(self):
        return {k: getattr(self, k) for k in self._train_keys()}

    def _set_train_state(self, d):
        for k in self._train_keys():
            if k in d:
                setattr(self, k, d[k])

    # RNG helpers
    def _collect_rng(self):
        # example:
        # return {
        #     "python": random.getstate(),
        #     "numpy": np.random.get_state(),
        #     "torch": torch.get_rng_state(),
        #     "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        # }
        pass

    def _restore_rng(self, state):
        # reverse of _collect_rng
        pass

    # ---- main API ----
    def state_dict(self):
        return {
            "agent": self.agent.state_dict(),
            "rng_state": self._collect_rng(),
            "train_state": self._get_train_state(),
            "config": {
                "rl_config": dataclasses.asdict(self.rl_config),
                "model_config": dataclasses.asdict(self.model_config),
            },
        }

    def load_state_dict(self, ckpt):
        self.agent.load_state_dict(ckpt["agent"])
        self._restore_rng(ckpt["rng_state"])
        self._set_train_state(ckpt.get("train_state", {}))
        # config is optional to actually use on load

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu"):
        raw = torch.load(path, map_location=map_location)
        self.load_state_dict(raw)


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
