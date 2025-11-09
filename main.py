import gymnasium as gym
from agent import Agent, PPOMemory
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
from collections import deque
from hyperparams import config

model_recent_path = "saved_models/ppo_flappy_bird_recent.pth"
model_best_path = "saved_models/ppo_flappy_bird_best.pth"
tensorboard_log_dir = "runs/ppo_experiment_hparams2"

run_mode = "test"  # "train" or "test"

if __name__ == '__main__':
    
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    # env = gym.make("FlappyBird-v0", use_lidar=True)
    agent = Agent(config)
    try:
        agent.load_models(model_best_path, config)
    except:
        print("No saved models found, starting fresh.")
    memory = PPOMemory(batch_size=config.batch_size)
    writer = SummaryWriter(tensorboard_log_dir)

    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)
    t = 0
    terminated = False
    rew_sum = 0.0
    total_games = 0
    rewards_fifo = deque(maxlen=100)
    max_high_score = 0
    avg_rew = 0.0
    max_avg_rew = agent.max_avg_rew

    while t < config.total_timesteps:
        
        for _ in range(config.rollout_len):
            obs_t = obs_t1
            value_t = agent.critic.forward(obs_t).item()
            action_dist = agent.actor.forward(obs_t)
            if run_mode == "train":    
                action = action_dist.sample()
            else:
                action = torch.argmax(action_dist.probs)
            action_t = int(action.item())
            log_prob_t = action_dist.log_prob(action).item()
            
            # Step the environment
            obs_t1, reward, terminated, truncated, info = env.step(action_t)
            obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)            

            done_t = terminated
            needs_reset = terminated or truncated
            reward_t = reward
            rew_sum += reward_t
            memory.store_memory(obs_t, log_prob_t, value_t, reward_t, action_t, done_t)
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
                        agent.save_models(model_best_path, config, max_avg_rew) 
                writer.add_scalar("Avg Episode Reward (last 100 games)", avg_rew, total_games)
                writer.add_scalar("Max High Score", max_high_score, total_games)
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype).to(agent.device)
            t += 1
            print(f"{value_t:.2f}\t,{reward_t:.2f},\t{done_t},\t {action_t},\t\
                    {log_prob_t:.2f}\t {info},\t {avg_rew:.2f}")
        
        # We need one more state and value to compute the last advantage
        if run_mode == "train":
            value_t1 = agent.critic.forward(obs_t1)
            memory.add_last_obs_value(obs_t1, value_t1.item())
            agent.learn(memory, writer, t//config.rollout_len)
            memory.clear_memory()
            agent.save_models(model_recent_path, config, max_avg_rew)
    
    writer.close()
    env.close()
