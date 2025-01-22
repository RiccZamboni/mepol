python src/experiments/mamepol.py --env "HandReach" \
    --k 4  --kl_threshold 15 --max_off_iters 30 --learning_rate 0.00001 \
    --num_trajectories 20 --trajectory_length 50 --num_epochs 10000 --heatmap_every 1000 \
    --heatmap_episodes 20 --heatmap_num_steps 50 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 2 --full_entropy_k 4 --num_workers 2 --update_algo "Decentralized" --policy_decentralized 1 --beta 0.1