import gymnasium as gym
from agent import Agent, PPOMemory
import torch
from torch.utils.tensorboard import SummaryWriter
import flappy_bird_gymnasium
from collections import deque
from configuration import ppo_config, mlp_config, transformer_config, make_hparams_dict
from checkpoint import CheckpointManager, make_paths
import os
from PIL import Image
from pygame_viewer import PygameViewer
from value_overlay import ValueOverlayConfig, ValueRewardOverlay
import argparse
import dataclasses

# Constants and defaults
DEFAULT_MODEL_CONFIG = mlp_config
# DEFAULT_MODEL_CONFIG = transformer_config
RL_CONFIG = ppo_config
ATTENTION_DEBUG_DIR = "debug_frames"
ATTENTION_DEBUG_LIMIT = 0  # Set to >0 to enable attention visualization in eval

VIEW_FPS = 30

def parse_args():
    parser = argparse.ArgumentParser(description="Flappy Bird PPO (train/eval) with optional value overlay UI.")
    parser.add_argument("--mode", choices=["train", "eval"], default="train", help="Run mode (default: train).")
    parser.add_argument(
        "--show-value",
        action="store_true",
        help="Show gameplay with a rolling value-function overlay (default: off).",
    )
    parser.add_argument("--human-play", action="store_true", help="In eval, control the bird yourself (Space/Up flap).")
    parser.add_argument(
        "--model",
        choices=["mlp", "transformer", "xfmr"],
        default=None,
        help="Override model type (default: uses DEFAULT_MODEL_CONFIG).",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=None,
        help="Override model dimension (MLP: hidden size; Transformer: embedding size).",
    )
    parser.add_argument(
        "--d_model_critic",
        type=int,
        default=None,
        help="(MLP only) Override critic hidden size (default: matches --d_model when provided).",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=None,
        help="(Transformer only) Override feedforward size (default: 4 * d_model when --d_model is provided).",
    )
    return parser.parse_args()

def build_model_config(args):
    model_name = args.model
    if model_name is None:
        base = DEFAULT_MODEL_CONFIG
        model_name = "transformer" if getattr(base, "model_type", None) == "xfmr" else "mlp"
    else:
        model_name = "transformer" if model_name in ("transformer", "xfmr") else "mlp"
        base = transformer_config if model_name == "transformer" else mlp_config

    if args.d_model is None and args.d_model_critic is None and args.mlp_dim is None:
        return base

    if model_name == "transformer":
        d_model = args.d_model if args.d_model is not None else base.d_model
        mlp_dim = args.mlp_dim if args.mlp_dim is not None else (d_model * 4 if args.d_model is not None else base.mlp_dim)
        return dataclasses.replace(base, d_model=d_model, mlp_dim=mlp_dim)

    d_model_actor = args.d_model if args.d_model is not None else base.d_model
    d_model_critic = args.d_model_critic if args.d_model_critic is not None else (d_model_actor if args.d_model is not None else base.d_model_critic)
    return dataclasses.replace(base, d_model=d_model_actor, d_model_critic=d_model_critic)

def setup_env(render_mode=None):
    if render_mode is None:
        return gym.make("FlappyBird-v0", use_lidar=True)
    return gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)

def init_obs_buffer(seq_len, obs_size=180, dtype=torch.float32):
    return deque([torch.ones(obs_size, dtype=dtype)] * (seq_len - 1), maxlen=seq_len)

def eval_loop(env, agent, model_config, args, paths):
    if not args.show_value:
        raise SystemExit("Eval mode requires `--show-value` (otherwise there is no window/controls).")

    human_play = args.human_play
    viewer = PygameViewer(window_scale=1, target_fps=VIEW_FPS)
    overlay = ValueRewardOverlay(ValueOverlayConfig())
    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
    obs_buffer = init_obs_buffer(model_config.seq_len, dtype=agent.dtype)
    obs_buffer.append(obs_t1)
    env_step_count = 0
    attention_captures = 0

    print("[eval] Controls: Space/Up=flap  R=reset  P=pause  Q/Esc=quit")
    try:
        # Render an initial frame so controls work immediately.
        with torch.no_grad():
            _, value_0 = agent.forward(torch.stack(list(obs_buffer), dim=0).unsqueeze(0).to(agent.device))
        frame = env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None; `--show-value` requires Gym render_mode='rgb_array'.")
        overlay.update(float(value_0.item()), 0.0)
        dt = viewer.show(overlay.draw(frame))
        overlay.update_fps(dt_sec=dt)

        while True:
            controls = viewer.poll()
            if controls.quit:
                break
            if controls.reset:
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                obs_buffer = init_obs_buffer(model_config.seq_len, dtype=agent.dtype)
                obs_buffer.append(obs_t1)
                overlay.reset()
                continue
            if viewer.paused:
                dt = viewer.tick()
                overlay.update_fps(dt_sec=dt)
                continue

            with torch.no_grad():
                action_dist, _ = agent.forward(torch.stack(list(obs_buffer), dim=0).unsqueeze(0).to(agent.device))

            action_t = 1 if (human_play and controls.flap) else int(torch.argmax(action_dist.probs).item())
            obs_t1, reward, terminated, truncated, info = env.step(action_t)
            obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
            obs_buffer.append(obs_t1)
            env_step_count += 1

            with torch.no_grad():
                _, value_t = agent.forward(torch.stack(list(obs_buffer), dim=0).unsqueeze(0).to(agent.device))

            frame = env.render()
            if frame is None:
                raise RuntimeError("env.render() returned None; `--show-value` requires Gym render_mode='rgb_array'.")

            overlay.update(float(value_t.item()), float(reward))
            if attention_captures < ATTENTION_DEBUG_LIMIT and hasattr(agent.model, "visualize_attention"):
                os.makedirs(ATTENTION_DEBUG_DIR, exist_ok=True)
                save_path = os.path.join(ATTENTION_DEBUG_DIR, f"attention_step_{env_step_count:05d}.png")
                agent.model.visualize_attention(save_path=save_path, env_frame=frame, frame_title=f"Step {env_step_count}")
                attention_captures += 1

            dt = viewer.show(overlay.draw(frame))
            overlay.update_fps(dt_sec=dt)

            if terminated or truncated:
                obs_t1, _ = env.reset()
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                obs_buffer = init_obs_buffer(model_config.seq_len, dtype=agent.dtype)
                obs_buffer.append(obs_t1)
                overlay.reset()
    finally:
        viewer.close()

def train_loop(env, agent, memory, writer, ckpt, model_config, rl_config, args, paths):
    show_value = args.show_value
    viewer = PygameViewer(window_scale=1, target_fps=VIEW_FPS) if show_value else None
    overlay = ValueRewardOverlay(ValueOverlayConfig()) if show_value else None
    obs_buffer = init_obs_buffer(model_config.seq_len, dtype=agent.dtype)
    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
    if not isinstance(ckpt.rewards_fifo, deque):
        ckpt.rewards_fifo = deque(ckpt.rewards_fifo, maxlen=10000)
    if not isinstance(ckpt.scores_queue, deque):
        ckpt.scores_queue = deque(ckpt.scores_queue, maxlen=10000)
    rewards_fifo = ckpt.rewards_fifo
    scores_queue = ckpt.scores_queue
    env_step_count = 0

    if viewer is not None:
        print("[train] Controls: P=pause  Q/Esc=quit")
        first_frame = env.render()
        if first_frame is None:
            raise RuntimeError("env.render() returned None; `--show-value` requires Gym render_mode='rgb_array'.")
        viewer.show(overlay.draw(first_frame), tick=False)
    try:
        while ckpt.training_step < rl_config.total_train_steps:
            for _ in range(rl_config.rollout_len):
                env_step_count += 1
                obs_buffer.append(obs_t1)
                action_dist, value_t = agent.forward(torch.stack(list(obs_buffer), dim=0).unsqueeze(0).to(agent.device))
                value_t = value_t.item()
                action = action_dist.sample()
                action_t = int(action.item())
                log_prob_t = action_dist.log_prob(action).item()

                obs_t1, reward_t, terminated, truncated, info = env.step(action_t)
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                done_t = terminated

                memory.store_memory(list(obs_buffer), log_prob_t, value_t, reward_t, action_t, done_t)
                ckpt.max_high_score = max(info.get("score", 0), ckpt.max_high_score)
                rewards_fifo.append(reward_t)

                if viewer is not None:
                    frame = env.render()
                    if frame is None:
                        raise RuntimeError("env.render() returned None; `--show-value` requires Gym render_mode='rgb_array'.")
                    overlay.update(float(value_t), float(reward_t))
                    dt = viewer.show(overlay.draw(frame))
                    overlay.update_fps(dt_sec=dt)

                    controls = viewer.poll()
                    if controls.quit:
                        ckpt.save(paths["checkpoint_recent"])
                        raise SystemExit(0)
                    while viewer.paused:
                        viewer.show(overlay.draw(frame))
                        controls = viewer.poll()
                        if controls.quit:
                            ckpt.save(paths["checkpoint_recent"])
                            raise SystemExit(0)

                if terminated or truncated:
                    if terminated:
                        score = info.get("score", 0)
                        scores_queue.append(score)
                        ckpt.last_score = score
                    obs_t1, _ = env.reset()
                    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                    obs_buffer = init_obs_buffer(model_config.seq_len, dtype=agent.dtype)
                    if overlay is not None:
                        overlay.reset()

            # Compute last advantage
            obs_buffer.append(obs_t1)
            _, value_t1 = agent.forward(torch.stack(list(obs_buffer), dim=0).unsqueeze(0).to(agent.device))
            memory.add_last_obs_value(list(obs_buffer), value_t1.item())

            # Learn and log
            agent.learn(memory, writer, ckpt.training_step)
            ckpt.training_step += 1
            memory.clear_memory()

            last_1k_rew = sum(list(rewards_fifo)[-1000:])
            if last_1k_rew > ckpt.max_1k_rew:
                ckpt.max_1k_rew = last_1k_rew
                ckpt.save(paths["checkpoint_best"])
            if ckpt.training_step % 5 == 0:
                ckpt.save(paths["checkpoint_recent"])
            writer.add_scalar("Total 1000 Step Reward", last_1k_rew, global_step=ckpt.training_step)
            writer.add_scalar("Max High Score", ckpt.max_high_score, global_step=ckpt.training_step)
            writer.add_scalar("Last Score", ckpt.last_score, global_step=ckpt.training_step)
            if scores_queue:
                writer.add_scalar("Mean Score", sum(scores_queue) / len(scores_queue), global_step=ckpt.training_step)
            print(f"Step {ckpt.training_step}: Value {value_t:.2f}, Reward {reward_t:.2f}, Done {done_t}, Action {action_t}, Log prob {log_prob_t:.2f}, Info {info}, Last 1k Rew {last_1k_rew:.2f}")
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()

if __name__ == '__main__':
    args = parse_args()
    run_mode = args.mode
    model_config = build_model_config(args)
    render_mode = "rgb_array" if args.show_value else None
    env = setup_env(render_mode=render_mode)
    agent = Agent(RL_CONFIG, model_config)
    paths = make_paths(agent, "flappy_bird", RL_CONFIG.rl_type)
    ckpt = CheckpointManager(agent, RL_CONFIG)
    memory = PPOMemory(batch_size=model_config.batch_size)
    writer = SummaryWriter(paths["tensorboard_dir"]) if run_mode == "train" else None
    if writer:
        writer.add_text("hparams", str(make_hparams_dict(RL_CONFIG, model_config)))

    if os.path.exists(paths["checkpoint_recent"]):
        ckpt.load(paths["checkpoint_recent"], map_location=agent.device)
    agent.model.train() if run_mode == "train" else agent.model.eval()

    try:
        if run_mode == "eval":
            eval_loop(env, agent, model_config, args, paths)
        else:
            train_loop(env, agent, memory, writer, ckpt, model_config, RL_CONFIG, args, paths)
    finally:
        env.close()
