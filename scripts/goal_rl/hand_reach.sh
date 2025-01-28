python src/experiments/goal_rl.py --env "HandReach" \
    --num_epochs 1000 --batch_size 20 --traj_len 25 \
    --critic_batch_size 64 --critic_iters 5 \
    --cg_iters 10 --kl_thresh 0.001 