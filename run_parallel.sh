#!/usr/bin/env bash
set -euo pipefail

REPO="/home/geoff/code/flappy-bird-rl"
ACTIVATE_CMD="source .venv/bin/activate"
PY_CMD="python3"

cd "$REPO"
# shellcheck disable=SC1091
eval "$ACTIVATE_CMD"

mkdir -p runs/flappy_bird_ppo_xfmr_dim32
mkdir -p runs/flappy_bird_ppo_xfmr_dim128
mkdir -p runs/flappy_bird_ppo_mlp_dim256
mkdir -p runs/flappy_bird_ppo_mlp_dim2048

$PY_CMD main.py --model transformer --d_model 32  |& tee runs/flappy_bird_ppo_xfmr_dim32/console.log &
$PY_CMD main.py --model transformer --d_model 128 |& tee runs/flappy_bird_ppo_xfmr_dim128/console.log &
$PY_CMD main.py --model mlp --d_model 256         |& tee runs/flappy_bird_ppo_mlp_dim256/console.log &
$PY_CMD main.py --model mlp --d_model 2048        |& tee runs/flappy_bird_ppo_mlp_dim2048/console.log &

wait
