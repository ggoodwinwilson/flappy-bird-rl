import gymnasium as gym
from agent import Agent, PPOMemory
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
from collections import deque
from configuration import ppo_config, mlp_config, transformer_config
from checkpoint import Checkpoint, CheckpointManager, make_paths

run_mode = "train"
# run_mode = "eval"
# model_config = transformer_config
model_config = mlp_config
rl_config = ppo_config

if __name__ == '__main__':
    
    # env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    env = gym.make("FlappyBird-v0", use_lidar=True)
    
    # Model is defined in the Agent class
    agent = Agent(rl_config, model_config)
    ckpt = CheckpointManager(agent, rl_config)
    paths = make_paths(model_config, "flappy_bird", rl_config.rl_type)
    agent.load(paths["checkpoint_recent"])
    memory = PPOMemory(batch_size=model_config.batch_size)
    writer = SummaryWriter(paths["tensorboard_dir"])

    # Initialize the first 9 observations in the episode for xfmr model
    obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
    
    training_steps = 0
    terminated = False
    rewards_fifo = deque(maxlen=10000)
    max_high_score = 0
    max_avg_rew = 0.0

    while training_steps < rl_config.total_train_steps:
        
        for _ in range(rl_config.rollout_len):
            obs_t = obs_t1
            obs_buffer.append(obs_t)
            action_dist, value_t = agent.forward(torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device))
            value_t = value_t.item()
            if run_mode == "train":
                action = action_dist.sample()
            else:
                action = torch.argmax(action_dist.probs)
            action_t = int(action.item())
            log_prob_t = action_dist.log_prob(action).item()
            
            # Step the environment
            obs_t1, reward, terminated, truncated, info = env.step(action_t)
            obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)           

            done_t = terminated
            needs_reset = terminated or truncated
            reward_t = reward
            if run_mode == "train":
                memory.store_memory(obs_buffer[-10:], log_prob_t, value_t, reward_t, action_t, done_t)
            max_high_score = max(info.get("score"), max_high_score)

            rewards_fifo.append(reward_t)

            # If done, start a new game
            if needs_reset:
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
            print(
                f"Value: {value_t:.2f},\tReward: {reward_t:.2f},\tDone: {done_t},\t"
                f"Action: {action_t},\tLog prob: {log_prob_t:.2f}\t, Info: {info},\t Avg Reward: {avg_rew:.2f}"
            )
        
        if run_mode == "train":
            
            # We need one more state and value to compute the last advantage
            obs_buffer.append(obs_t1)
            _, value_t1 = agent.forward(torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device))
            memory.add_last_obs_value(obs_buffer[-10:], value_t1.item())
            
            # Train the model
            agent.learn(memory, writer, training_steps)
            training_steps += 1
            memory.clear_memory()
            
            # Log training statsand save models
            avg_rew = sum(rewards_fifo)/len(rewards_fifo)
            writer.add_scalar("Total 1000 Step Reward", avg_rew, global_step=training_steps)
            writer.add_scalar("Max High Score", max_high_score, global_step=training_steps)
            if avg_rew > max_avg_rew:
                max_avg_rew = avg_rew
                agent.save_models(paths["checkpoint_best"]) 
            if training_steps % 5 == 0:
                agent.save_models(paths["checkpoint_recent"])
    
    writer.close()
    env.close()
