python src/experiments/mamepol.py --env "Push_Box" \
    --k 4  --kl_threshold 15 --max_off_iters 30 --learning_rate 0.00001 \
    --num_trajectories 20 --trajectory_length 200 --num_epochs 2000 --heatmap_every 100 \
    --heatmap_episodes 100 --heatmap_num_steps 200 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 5 --full_entropy_k 4 --num_workers 1 --update_algo "Decentralized_MI"