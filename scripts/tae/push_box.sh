python src/experiments/mamepol.py --env "Push_Box"\
    --k 4  --kl_threshold 6 --max_off_iters 20 --learning_rate 0.00001 \
    --num_trajectories 20 --trajectory_length 350 --num_epochs 10000 --heatmap_every 2000 \
    --heatmap_episodes 20 --heatmap_num_steps 350 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 5 --full_entropy_k 4 --num_workers 2 --update_algo "Centralized_MI" --policy_decentralized 1 --beta 0.1