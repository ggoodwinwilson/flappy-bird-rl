@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Launch 4 parallel training runs in one Windows Terminal window (4 tabs).
REM Edit these if your setup differs.
set "WSL_DISTRO="
set "REPO_WSL=/home/geoff/code/flappy-bird-rl"
set "ACTIVATE_CMD=source .venv/bin/activate"
set "PY_CMD=python3"

set "WSL_CMD=wsl"
if not "%WSL_DISTRO%"=="" set "WSL_CMD=wsl -d %WSL_DISTRO%"

wt ^
  new-tab --title "xfmr d=32"  %WSL_CMD% bash -lc "cd '%REPO_WSL%' && mkdir -p 'runs/flappy_bird_ppo_xfmr_dim32'   && %ACTIVATE_CMD% && %PY_CMD% main.py --model transformer --d_model 32  |& tee 'runs/flappy_bird_ppo_xfmr_dim32/console.log';   exec bash" ^
; new-tab --title "xfmr d=128" %WSL_CMD% bash -lc "cd '%REPO_WSL%' && mkdir -p 'runs/flappy_bird_ppo_xfmr_dim128'  && %ACTIVATE_CMD% && %PY_CMD% main.py --model transformer --d_model 128 |& tee 'runs/flappy_bird_ppo_xfmr_dim128/console.log';  exec bash" ^
; new-tab --title "mlp d=256"  %WSL_CMD% bash -lc "cd '%REPO_WSL%' && mkdir -p 'runs/flappy_bird_ppo_mlp_dim256'    && %ACTIVATE_CMD% && %PY_CMD% main.py --model mlp --d_model 256         |& tee 'runs/flappy_bird_ppo_mlp_dim256/console.log';    exec bash" ^
; new-tab --title "mlp d=2048" %WSL_CMD% bash -lc "cd '%REPO_WSL%' && mkdir -p 'runs/flappy_bird_ppo_mlp_dim2048'   && %ACTIVATE_CMD% && %PY_CMD% main.py --model mlp --d_model 2048        |& tee 'runs/flappy_bird_ppo_mlp_dim2048/console.log';   exec bash"
