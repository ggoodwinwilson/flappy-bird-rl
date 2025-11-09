from torch import nn, optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from hyperparams import PPOConfig

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

class ActorNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, observation):
        logits = self.net(observation)
        output_dist = torch.distributions.Categorical(logits=logits)
        return output_dist 
    
class CriticNetwork(nn.Module):
    def __init__(self, d_in, d_out, d_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out)
        )
        
    def forward(self, observation):
        return self.net(observation)
    
class Agent:
    def __init__(self, config: PPOConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(d_in=config.d_in, d_hidden=config.d_hidden_actor, 
                                  d_out=config.d_out_actor).to(self.device)
        self.critic = CriticNetwork(d_in=config.d_in, d_hidden=config.d_hidden_critic, 
                                    d_out=config.d_out_critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.max_avg_rew = float('-inf')
        self.batch_size = config.batch_size
        self.rollout_len = config.rollout_len
        self.num_epochs = config.num_epochs
        self.gamma = config.gamma
        self.td_lambda = config.td_lambda
        self.eps_clip = config.eps_clip
        self.dtype = config.dtype
        self.ent_coef = config.ent_coef
        self.critic_coef = config.critic_coef

    def learn(self, memory:PPOMemory, writer:SummaryWriter, traj_step):
        
        rewards = torch.as_tensor(memory.rewards, device=self.device, dtype=self.dtype)
        dones = torch.as_tensor(memory.dones, device=self.device, dtype=self.dtype)
        old_log_probs = torch.as_tensor(memory.log_probs, device=self.device, dtype=self.dtype)
        values = torch.as_tensor(memory.vals, device=self.device, dtype=self.dtype)
        observations = torch.stack(memory.obs, dim=0).to(self.device)
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

                # Sample a new action from the current policy
                new_action_dist = self.actor.forward(obs_batch)
                new_log_prob_batch = new_action_dist.log_prob(old_act_batch)
                entropy = new_action_dist.entropy().mean()

                # Compute actor loss function
                ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
                surr1 = ratio * adv_norm_batch
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_norm_batch
                # actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

                # Compute critic loss function
                new_value_batch = self.critic.forward(obs_batch).squeeze(-1)
                critic_loss = nn.MSELoss()(new_value_batch, rtg_batch)

                # Backprop total loss
                total_loss = actor_loss + self.critic_coef * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                # Gradient clipping for stability
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                step = epoch * (self.rollout_len//self.batch_size) + batch_num
                writer.add_scalar("Loss/policy", actor_loss.item(), traj_step*step)
                writer.add_scalar("Loss/value", critic_loss.item(), traj_step*step)
                writer.add_scalar("Loss/total", total_loss.item(), traj_step*step)

    def save_models(self, path, config, max_avg_rew):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
            "config": config.as_dict(),
            "max_avg_rew": max_avg_rew
        }, path)

    def load_models(self, path, config) -> float:
        bundle = torch.load(path, map_location=self.device)
        if bundle["config"] != config.as_dict():
            raise ValueError("Checkpoint config mismatch")
        self.actor.load_state_dict(bundle["actor"])
        self.critic.load_state_dict(bundle["critic"])
        self.actor_optimizer.load_state_dict(bundle["actor_opt"])
        self.critic_optimizer.load_state_dict(bundle["critic_opt"])
        self.max_avg_rew = bundle.get("max_avg_rew", float('-inf'))
