python src/experiments/goal_rl.py --env "Room" \
    --num_epochs 100 --batch_size 20 --traj_len 20 \
    --critic_batch_size 64 --critic_iters 5 \
    --cg_iters 10 --kl_thresh 0.001 --policy_init "room/Centralized_MI/T50_1/"