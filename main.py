import gymnasium as gym
import numpy as np
from agent import Agent, PPOMemory
# from utils import plot_learning_curve
import gymnasium as gym
import torch

# Hyperparameters
td_lambda = 0.95
gamma = 0.99
eps_clip = 0.2
learning_rate = 0.0003
num_epochs = 10
batch_size = 5
rollout_len = 20
total_timesteps = 10000
d_in = 180
d_out_actor = 2
d_out_critic = 1
d_hidden_actor = 128
d_hidden_critic = 128

if __name__ == '__main__':
    
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    agent = Agent(td_lambda=td_lambda, gamma=gamma, eps_clip=eps_clip, 
                  learning_rate=learning_rate, batch_size=batch_size, 
                  rollout_len=rollout_len, num_epochs=num_epochs,
                  d_in=d_in, d_out_actor=d_out_actor, d_out_critic=d_out_critic,
                  d_hidden_actor=d_hidden_actor, d_hidden_critic=d_hidden_critic)
    memory = PPOMemory(batch_size=batch_size)

    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)
    t = 0
    terminated = False

    while t < total_timesteps:
        for _ in range(rollout_len):
            obs_t = obs_t1
            value_t = agent.critic.forward(obs_t).item()
            action_dist = agent.actor.forward(obs_t)
            action = action_dist.sample()
            action_t = int(action.item())
            log_prob_t = action_dist.log_prob(action).item()
            
            # Step the environment
            obs_t1, reward, terminated, _, info = env.step(action_t)
            obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)            

            done_t = terminated
            reward_t = reward
            memory.store_memory(obs_t, log_prob_t, value_t, reward_t, action_t, done_t)

            # If done, start a new game
            if terminated:
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)
        
        # We need one more state to compute the last advantage
        value_t1 = agent.critic.forward(obs_t1)
        memory.append_obs_value(obs_t1, value_t1)
        
        agent.learn(memory)
        t += 1
    env.close()
