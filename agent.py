import flappy_bird_gymnasium
import gymnasium
from torch import nn, optim
import torch
import numpy as np

class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.next_obs = None
        self.next_val = None

    def store_memory(self, obs, log_prob, val, reward, action, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def append_obs_value(self, obs, val):
        self.obs.append(obs)
        self.vals.append(val)

    def clear_memory(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

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
    def __init__(self, td_lambda, gamma, eps_clip, learning_rate, 
                 batch_size, rollout_len, num_epochs,  d_in, 
                 d_out_actor, d_out_critic,
                  d_hidden_actor, d_hidden_critic):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(d_in=d_in, d_hidden=d_hidden_actor, 
                                  d_out=d_out_actor).to(self.device)
        self.critic = CriticNetwork(d_in=d_in, d_hidden=d_hidden_critic, 
                                    d_out=d_out_critic).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.rollout_len = rollout_len
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.eps_clip = eps_clip
        self.dtype = torch.float32

    def learn(self, memory:PPOMemory):
        
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

        # Normalize advantages for training stability
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(self.num_epochs):

            # Generate random minibatch indices
            indices = np.random.permutation(self.rollout_len)

            for start in range(0, self.rollout_len, self.batch_size):
                end = start + self.batch_size

                batch_indices = indices[start:end]

                old_act_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                old_value_batch = values[batch_indices]
                # obs_batch = torch.stack(observations[batch_indices], dim=0)
                obs_batch = observations[batch_indices]
                adv_batch = advantage[batch_indices]

                # Sample a new action from the current policy
                new_action_dist = self.actor.forward(obs_batch)
                new_log_prob_batch = new_action_dist.log_prob(old_act_batch)

                # Compute actor loss function
                ratio = torch.exp(new_log_prob_batch - old_log_prob_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # Compute critic loss function
                rtg_batch = adv_batch + old_value_batch
                new_value_batch = self.critic.forward(obs_batch)
                critic_loss = nn.MSELoss()(new_value_batch, rtg_batch)

                # Backprop total loss
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
