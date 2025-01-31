python src/experiments/goal_rl.py --env "HandReach" \
    --num_epochs 200 --batch_size 10 --traj_len 100 \
    --critic_batch_size 64 --critic_iters 5 \
    --cg_iters 10 --kl_thresh 0.001  --policy_init "handreach/Decentralized_KL/T100/"