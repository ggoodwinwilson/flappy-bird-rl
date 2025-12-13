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

Train while showing the game + a rolling value-function overlay:

```bash
python main.py --render --overlay
```

Tune rendering cost during training:

```bash
python main.py --render --overlay --render-every 4 --display-fps 60 --display-scale 1
```

### Eval

Watch the agent play with the value overlay:

```bash
python main.py --mode eval --render --overlay
```

Play manually in eval (Space/Up = flap):

```bash
python main.py --mode eval --render --overlay --human-play
```

## Viewer controls

When the pygame viewer window is open:

- `Space` / `Up`: flap (only used when `--human-play` is set)
- `R`: reset episode
- `P`: pause/unpause
- `Q` / `Esc`: quit

## Notes

- The on-screen overlay needs pixel frames, so rendering uses Gym `render_mode="rgb_array"` and the UI is displayed in a separate pygame window.
