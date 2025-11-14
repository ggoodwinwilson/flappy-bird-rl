import gymnasium as gym
from agent import Agent, PPOMemory
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
from collections import deque
from configuration import ppo_config, mlp_config, transformer_config

model_recent_path = "saved_models/ppo_flappy_bird_recent_mlp.pth"
model_best_path = "saved_models/ppo_flappy_bird_best_mlp.pth"
tensorboard_log_dir = "runs/ppo_mlp"

run_mode = "train"  # "train" or "test"
# model_config = transformer_config
model_config = mlp_config

if __name__ == '__main__':
    
    # env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    env = gym.make("FlappyBird-v0", use_lidar=True)
    agent = Agent(ppo_config, model_config)
    try:
        agent.load_models(model_best_path)
    except:
        print("No saved models found, starting fresh.")
    memory = PPOMemory(batch_size=model_config.batch_size)
    writer = SummaryWriter(tensorboard_log_dir)

    obs_buffer = []
    if model_config.model_type == "transformer":
        # Initialize the first 9 observations in the episode
        obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
    
    t = 0
    terminated = False
    rew_sum = 0.0
    total_games = 0
    rewards_fifo = deque(maxlen=100)
    max_high_score = 0
    avg_rew = 0.0
    max_avg_rew = agent.max_avg_rew

    while t < ppo_config.total_timesteps:
        
        for _ in range(ppo_config.rollout_len):
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
            rew_sum += reward_t
            if run_mode == "train":
                memory.store_memory(obs_buffer[-10:], log_prob_t, value_t, reward_t, action_t, done_t)
            max_high_score = max(info.get("score"), max_high_score)

            # If done, start a new game
            if needs_reset:
                total_games += 1
                rewards_fifo.append(rew_sum)
                rew_sum = 0.0
                avg_rew = sum(rewards_fifo)/len(rewards_fifo)
                if avg_rew > max_avg_rew:
                    max_avg_rew = avg_rew
                    if run_mode == "train":
                        agent.save_models(model_best_path, max_avg_rew) 
                writer.add_scalar("Avg Episode Reward (last 100 games)", avg_rew, total_games)
                writer.add_scalar("Max High Score", max_high_score, total_games)
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                if model_config.model_type == "transformer":
                    obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
            t += 1
            print(
                f"Value: {value_t:.2f},\tReward: {reward_t:.2f},\tDone: {done_t},\t"
                f"Action: {action_t},\tLog prob: {log_prob_t:.2f}\t, Info: {info},\t Avg Reward: {avg_rew:.2f}"
            )
        
        # We need one more state and value to compute the last advantage
        if run_mode == "train":
            obs_buffer.append(obs_t1)
            _, value_t1 = agent.forward(torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device))
            memory.add_last_obs_value(obs_buffer[-10:], value_t1.item())
            agent.learn(memory, writer, t//ppo_config.rollout_len)
            memory.clear_memory()
            agent.save_models(model_recent_path, max_avg_rew)
    
    writer.close()
    env.close()
