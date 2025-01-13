#!/bin/bash
#
#SBATCH --job-name=marl-seppo-d2
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=7-00:00:00 # days-hh:mm:ss
#SBATCH --mem-per-cpu=6G
#SBATCH --output=output/%j.out
#SBATCH --gres=gpu:1

#SBATCH --array=0-1


export WANDB_API_KEY=eff49591de386b20e04c17fb2456092bbc34cefe
# nb for running headless - add pwd to path
export PYTHONPATH="${PYTHONPATH}:`pwd`"

# Create a virtual display with OpenGL support
Xvfb :$SLURM_JOB_ID -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:$SLURM_JOB_ID

python src/experiments/mamepol.py --env "Room" \
    --k 4  --kl_threshold 6 --max_off_iters 20 --learning_rate 0.00001 \
    --num_trajectories 10 --trajectory_length 150 --num_epochs 10000 --heatmap_every 50 \
    --heatmap_episodes 100 --heatmap_num_steps 500 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 5 --full_entropy_k 4 --num_workers 2 --update_algo "Centralized" --policy_decentralized 1 --beta 0.1