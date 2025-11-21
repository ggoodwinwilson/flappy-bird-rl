from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from configuration import PPOConfig
from transformer import Transformer
from mlp import MLP

class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def store_memory(self, obs, log_prob, val, reward, action, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def add_last_obs_value(self, obs, val):
        self.obs.append(obs)
        self.vals.append(val)

    def clear_memory(self):
        del self.obs[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.vals[:]
        del self.rewards[:]
        del self.dones[:]
    
class Agent:
    def __init__(self, rl_config: PPOConfig, model_config):
        self.rl_config = rl_config
        self.rollout_len = self.rl_config.rollout_len
        self.gamma = self.rl_config.gamma
        self.td_lambda = self.rl_config.td_lambda
        self.eps_clip = self.rl_config.eps_clip
        self.ent_coef = self.rl_config.ent_coef
        self.critic_coef = self.rl_config.critic_coef
        self.model_config = model_config
        self.batch_size = self.model_config.batch_size
        self.num_epochs = self.model_config.num_epochs
        self.dtype = self.model_config.dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model_config.model_type == "xfmr":
            self.model = Transformer(config=self.model_config).to(self.device)
        else:
            self.model = MLP(config=self.model_config).to(self.device)

    def learn(self, memory:PPOMemory, writer:SummaryWriter, global_timestep:int):
        
        rewards = torch.as_tensor(memory.rewards, device=self.device, dtype=self.dtype)
        dones = torch.as_tensor(memory.dones, device=self.device, dtype=self.dtype)
        old_log_probs = torch.as_tensor(memory.log_probs, device=self.device, dtype=self.dtype)
        values = torch.as_tensor(memory.vals, device=self.device, dtype=self.dtype)
        if self.model_config.model_type == "xfmr":
            observations = torch.stack([torch.stack(obs, dim=0) for obs in memory.obs], dim=0).to(self.device)
        else:
            # Observations needs to be stacked and reshaped for MLP
            observations = [torch.stack(obs, dim=0) for obs in memory.obs]
            observations = [obs[-1:].clone() for obs in observations]
            observations = torch.stack(observations, dim=0).to(self.device)
        actions = torch.as_tensor(memory.actions, device=self.device, dtype=torch.int64)

        delta = torch.zeros(self.rollout_len, device=self.device, dtype=self.dtype)
        advantage = torch.zeros(self.rollout_len+1, device=self.device, dtype=self.dtype)
        rtg = torch.zeros(self.rollout_len, device=self.device, dtype=self.dtype)

        advantage[self.rollout_len] = 0

        for t in reversed(range(self.rollout_len)):
            # Compute temporal-difference first
            # δ_t = r_t + γ(1−done_t) V(s_{t+1}) − V(s_t)
            delta[t] = rewards[t] + self.gamma*(1-dones[t])*values[t+1] - values[t]

            # Comptue advantage using GAE
            # A_t = δ_t + γλ(1−done_t) A_{t+1}
            advantage[t] = delta[t] + self.gamma*self.td_lambda*(1-dones[t])*advantage[t+1]

            # Compute rewards-to-go
            # R_t = A_t + V(s_t)
            rtg[t] = advantage[t] + values[t]

        # Normalize advantages for training stability
        advantage = advantage[:-1]
        advantage_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for epoch in range(self.num_epochs):

            # Generate random minibatch indices
            indices = np.random.permutation(self.rollout_len)
            batch_num = 0

            for start in range(0, self.rollout_len, self.batch_size):
                batch_num += 1
                end = start + self.batch_size

                batch_indices = indices[start:end]

                old_act_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                obs_batch = observations[batch_indices]
                adv_norm_batch = advantage_norm[batch_indices]
                rtg_batch = rtg[batch_indices]

                # Sample a new actions and values from the current policy
                new_action_dist, new_value_batch = self.forward(obs_batch)
                new_log_prob_batch = new_action_dist.log_prob(old_act_batch)
                entropy = new_action_dist.entropy().mean()
                new_value_batch = new_value_batch.squeeze(-1)

                # Compute actor loss function
                ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
                surr1 = ratio * adv_norm_batch
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_norm_batch
                # actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy
                # actor_loss = -(ratio * adv_norm_batch).mean()

                # Compute critic loss function
                critic_loss = nn.MSELoss()(new_value_batch, rtg_batch)

                # Backprop total loss
                total_loss = actor_loss + self.critic_coef * critic_loss
                self.model.optim_zero_grad()
                total_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()),
                    max_norm=0.5
                )
                self.model.optim_step()

                # step = epoch * (self.rollout_len//self.batch_size) + batch_num
                # print(f"epoch: {epoch},\t go to 1: {new_action_dist.probs.gather(-1, old_act_batch.unsqueeze(-1)).mean():.4f},\t critic_loss: {critic_loss:.4f},\t entropy: {entropy:.4f},\t go to 1: {(new_action_dist.logits.argmax(dim=-1) == old_act_batch).float().mean():.4f}")

    def forward(self, x:list):
        if self.model_config.model_type == "xfmr":
            return self.model.forward(x)
        else:
            return self.model.forward(x[:,-1:,:].squeeze(1))
        
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "actor_opt": getattr(self.model, "actor_optimizer", None) and self.model.actor_optimizer.state_dict(),
            "critic_opt": getattr(self.model, "critic_optimizer", None) and self.model.critic_optimizer.state_dict(),
            "optimizer": getattr(self.model, "optimizer", None) and self.model.optimizer.state_dict(),
            "scheduler": getattr(self, "scheduler", None) and self.scheduler.state_dict(),
            "scaler": getattr(self, "scaler", None) and self.scaler.state_dict(),
            "config": {
                "ppo": self.rl_config.as_dict(),
                "model": self.model_config.as_dict(),
            },
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        if state.get("actor_opt") and hasattr(self.model, "actor_optimizer"):
            self.model.actor_optimizer.load_state_dict(state["actor_opt"])
        if state.get("critic_opt") and hasattr(self.model, "critic_optimizer"):
            self.model.critic_optimizer.load_state_dict(state["critic_opt"])
        if state.get("optimizer") and hasattr(self.model, "optimizer"):
            self.model.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") and hasattr(self, "scaler"):
            self.scaler.load_state_dict(state["scaler"])