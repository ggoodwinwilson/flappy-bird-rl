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
from value_overlay import ValueGraphOverlay, ValueOverlayConfig

run_mode = os.environ.get("RUN_MODE", "train").strip().lower()
if run_mode not in {"train", "eval"}:
    raise ValueError("RUN_MODE must be 'train' or 'eval'")
# model_config = transformer_config
model_config = mlp_config
rl_config = ppo_config
ATTENTION_DEBUG = run_mode == "eval"
ATTENTION_DEBUG_LIMIT = 0
ATTENTION_DEBUG_DIR = "debug_frames"

if __name__ == '__main__':
    
    # --- Eval/Render toggles ---
    # Uncomment to play manually (only meaningful in eval):
    # HUMAN_PLAY = True
    HUMAN_PLAY = False
    SHOW_VALUE_OVERLAY = True
    RENDER_DURING_TRAIN = True
    RENDER_EVERY_N_STEPS = 2  # increase for faster training
    # The value overlay needs pixel frames; we display them in our own pygame window.
    # If you set this to "human", the environment renders its own window and returns no frames (so no overlay).
    EVAL_RENDER_MODE = "rgb_array"
    DISPLAY_SCALE = 1
    DISPLAY_FPS = 60

    want_pixels = (run_mode == "eval") or (run_mode == "train" and RENDER_DURING_TRAIN)
    render_mode = (EVAL_RENDER_MODE if (run_mode == "eval") else "rgb_array") if want_pixels else None
    if want_pixels and SHOW_VALUE_OVERLAY and render_mode != "rgb_array":
        print("[warn] SHOW_VALUE_OVERLAY requires render_mode='rgb_array'; overriding.")
        render_mode = "rgb_array"
    if render_mode is None:
        env = gym.make("FlappyBird-v0", use_lidar=True)
    else:
        env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)
    
    # Model is defined in the Agent class
    agent = Agent(rl_config, model_config)
    ckpt = CheckpointManager(agent, rl_config)
    paths = make_paths(agent, "flappy_bird", rl_config.rl_type)
    memory = PPOMemory(batch_size=model_config.batch_size)
    writer = None
    if run_mode == "train":
        writer = SummaryWriter(paths["tensorboard_dir"])
        writer.add_text("hparams", str(make_hparams_dict(rl_config, model_config)))
    
    # Resume checkpoint if it exists
    if os.path.exists(paths["checkpoint_recent"]):
        ckpt.load(paths["checkpoint_recent"], map_location=agent.device)

    if run_mode == "train":
        agent.model.train()
    else:
        agent.model.eval()

    # Initialize the first 9 observations in the episode for xfmr model
    obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
    obs_t1, _ = env.reset()
    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)

    terminated = False
    rewards_fifo = deque(maxlen=10000)
    scores_queue = deque(maxlen=10000)
    max_high_score = 0
    ckpt.max_1k_rew = float('-inf')
    env_step_count = 0
    attention_captures = 0

    if run_mode == "eval":
        viewer = PygameViewer(window_scale=DISPLAY_SCALE, target_fps=DISPLAY_FPS)
        value_overlay = ValueGraphOverlay(ValueOverlayConfig()) if SHOW_VALUE_OVERLAY else None
        try:
            print("[eval] Controls: Space/Up=flap  R=reset  P=pause  Q/Esc=quit")
            while True:
                env_step_count += 1
                obs_t = obs_t1
                obs_buffer.append(obs_t)

                with torch.no_grad():
                    action_dist, value_t = agent.forward(
                        torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device)
                    )
                value_t = float(value_t.item())

                frame = env.render()
                if frame is None:
                    raise RuntimeError(
                        "env.render() returned None. For the overlay, run eval with render_mode='rgb_array' "
                        "(set EVAL_RENDER_MODE='rgb_array' and don't use render_mode='human')."
                    )
                if value_overlay is not None:
                    value_overlay.update(value_t)

                if (
                    ATTENTION_DEBUG
                    and attention_captures < ATTENTION_DEBUG_LIMIT
                    and hasattr(agent.model, "visualize_attention")
                ):
                    os.makedirs(ATTENTION_DEBUG_DIR, exist_ok=True)
                    save_path = os.path.join(
                        ATTENTION_DEBUG_DIR, f"attention_step_{env_step_count:05d}.png"
                    )
                    agent.model.visualize_attention(
                        save_path=save_path,
                        env_frame=frame,
                        frame_title=f"Step {env_step_count}",
                    )
                    attention_captures += 1

                dt = viewer.show(frame if value_overlay is None else value_overlay.draw(frame))
                if value_overlay is not None:
                    value_overlay.update_fps(dt_sec=dt)

                controls = viewer.poll()
                if controls.quit:
                    break
                if controls.reset:
                    obs_t1, _ = env.reset()
                    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                    obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
                    if value_overlay is not None:
                        value_overlay.reset()
                    continue
                if viewer.paused:
                    continue

                if HUMAN_PLAY:
                    action_t = 1 if controls.flap else 0
                else:
                    action_t = int(torch.argmax(action_dist.probs).item())

                obs_t1, reward, terminated, truncated, info = env.step(action_t)
                obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)

                needs_reset = terminated or truncated
                if needs_reset:
                    obs_t1, _ = env.reset()
                    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                    obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
                    if value_overlay is not None:
                        value_overlay.reset()
        finally:
            viewer.close()
    else:
        viewer = None
        value_overlay = None
        if RENDER_DURING_TRAIN:
            viewer = PygameViewer(window_scale=DISPLAY_SCALE, target_fps=DISPLAY_FPS)
            value_overlay = ValueGraphOverlay(ValueOverlayConfig()) if SHOW_VALUE_OVERLAY else None
            print("[train] Viewer: P=pause  Q/Esc=quit (training continues unless paused)")
        try:
            while ckpt.training_step < rl_config.total_train_steps:
                for _ in range(rl_config.rollout_len):
                    env_step_count += 1
                    obs_t = obs_t1
                    obs_buffer.append(obs_t)
                    action_dist, value_t = agent.forward(
                        torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device)
                    )
                    value_t = value_t.item()
                    action = action_dist.sample()
                    action_t = int(action.item())
                    log_prob_t = action_dist.log_prob(action).item()

                    # Step the environment
                    obs_t1, reward, terminated, truncated, info = env.step(action_t)
                    obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)

                    done_t = terminated
                    needs_reset = terminated or truncated
                    reward_t = reward
                    memory.store_memory(obs_buffer[-10:], log_prob_t, value_t, reward_t, action_t, done_t)
                    ckpt.max_high_score = max(info.get("score"), ckpt.max_high_score)

                    rewards_fifo.append(reward_t)

                    if viewer is not None and (env_step_count % max(1, RENDER_EVERY_N_STEPS) == 0):
                        frame = env.render()
                        if frame is None:
                            raise RuntimeError(
                                "env.render() returned None during training; set RENDER_DURING_TRAIN=False or ensure "
                                "render_mode='rgb_array'."
                            )
                        if value_overlay is not None:
                            value_overlay.update(float(value_t))
                        dt = viewer.show(frame if value_overlay is None else value_overlay.draw(frame))
                        if value_overlay is not None:
                            value_overlay.update_fps(dt_sec=dt)
                        controls = viewer.poll()
                        if controls.quit:
                            ckpt.save(paths["checkpoint_recent"])
                            raise SystemExit(0)
                        if viewer.paused:
                            while viewer.paused:
                                _dt = viewer.show(frame if value_overlay is None else value_overlay.draw(frame))
                                _controls = viewer.poll()
                                if _controls.quit:
                                    ckpt.save(paths["checkpoint_recent"])
                                    raise SystemExit(0)

                    # If done, start a new game
                    if needs_reset:
                        if terminated:
                            scores_queue.append(info.get("score", 0))
                        obs_t1, _ = env.reset()
                        obs_t1 = torch.as_tensor(obs_t1, dtype=agent.dtype)
                        obs_buffer = [torch.ones(180)] * (model_config.seq_len - 1)
                        if value_overlay is not None:
                            value_overlay.reset()

                # We need one more state and value to compute the last advantage
                obs_buffer.append(obs_t1)
                _, value_t1 = agent.forward(
                    torch.stack(obs_buffer[-10:], dim=0).unsqueeze(0).to(agent.device)
                )
                memory.add_last_obs_value(obs_buffer[-10:], value_t1.item())

                # Train the model
                agent.learn(memory, writer, ckpt.training_step)
                ckpt.training_step += 1
                memory.clear_memory()

                # Log training stats and save models
                last_1k_rew = sum(list(rewards_fifo)[-1000:])
                if last_1k_rew > ckpt.max_1k_rew:
                    ckpt.max_1k_rew = last_1k_rew
                    ckpt.save(paths["checkpoint_best"])
                if ckpt.training_step % 5 == 0:
                    ckpt.save(paths["checkpoint_recent"])
                writer.add_scalar("Total 1000 Step Reward", last_1k_rew, global_step=ckpt.training_step)
                writer.add_scalar("Max High Score", ckpt.max_high_score, global_step=ckpt.training_step)
                if len(scores_queue) > 0:
                    mean_score = sum(scores_queue) / len(scores_queue)
                    writer.add_scalar("Mean Score", mean_score, global_step=ckpt.training_step)
                print(
                    f"Value: {value_t:.2f},\tReward: {reward_t:.2f},\tDone: {done_t},\t"
                    f"Action: {action_t},\tLog prob: {log_prob_t:.2f}\t, Info: {info},\t Last 1k Rew: {last_1k_rew:.2f}"
                )
        finally:
            if writer is not None:
                writer.close()
            if viewer is not None:
                viewer.close()

    env.close()


def save_gamestate_snapshot(env, step_count, save_dir="debug_frames"):
    # 1. Get the raw numpy array (Height, Width, 3)
    frame = env.render()
    
    # 2. Convert to Image object
    image = Image.fromarray(frame)
    
    # 3. Save as JPG
    filename = f"{save_dir}/frame_{step_count:05d}.jpg"
    image.save(filename)
    # print(f"Saved {filename}")
