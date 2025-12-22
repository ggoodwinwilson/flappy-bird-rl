# Flappy Bird RL

This repo trains and evaluates a PPO agent on `FlappyBird-v0` (via `flappy_bird_gymnasium`).

## Running

All entry points are in `main.py`.

## Usage

See all flags/options:

```bash
python main.py --help
```

### Train (default)

Trains without rendering (fastest):

```bash
python main.py
```

### Model selection / overrides

By default, `main.py` uses `DEFAULT_MODEL_CONFIG` (see the top of `main.py`). You can override the model type and key dimensions from the CLI:

```bash
python main.py --model transformer --d_model 32
python main.py --model transformer --d_model 128
python main.py --model mlp --d_model 256
python main.py --model mlp --d_model 2048
```

- `--model`: `mlp` or `transformer` (alias: `xfmr`)
- `--d_model`: MLP hidden size (and default critic size), or Transformer embedding size
- `--d_model_critic`: (MLP only) critic hidden size override
- `--mlp_dim`: (Transformer only) feed-forward size override

Train while showing the game + value-function overlay (real-time):

```bash
python main.py --show-value
```

### Eval

Watch the agent play with the value overlay (real-time):

```bash
python main.py --mode eval --show-value
```

Play manually in eval (Space/Up = flap):

```bash
python main.py --mode eval --show-value --human-play
```

## Parallel runs (Windows + WSL)

If you're on Windows with WSL, `run_parallel.bat` launches one Windows Terminal window with 4 tabs and runs these in parallel:

- Transformer `d_model=32`
- Transformer `d_model=128`
- MLP `d_model=256`
- MLP `d_model=2048`

It reuses the same checkpoint + tensorboard directories as normal runs (so it will resume if checkpoints already exist).

From Windows (PowerShell/CMD):

```bat
.\run_parallel.bat
```

If needed, edit `WSL_DISTRO` and `REPO_WSL` at the top of `run_parallel.bat`.

## Viewer controls

When the pygame viewer window is open:

- `Space` / `Up`: flap (only used when `--human-play` is set)
- `R`: reset episode
- `P`: pause/unpause
- `Q` / `Esc`: quit

## Notes

- The on-screen overlay needs pixel frames, so rendering uses Gym `render_mode="rgb_array"` and the UI is displayed in a separate pygame window.
