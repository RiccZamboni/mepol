import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
import scipy
import scipy.special
import time
import os
from functools import reduce

from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from torch.utils import tensorboard

from src.utils.dtypes import float_type, int_type
from src.utils.convert2base import obs_to_int_pi, s_to_sp, convert_to_base


def get_heatmap(env, policies, states, num_traj, traj_len, cmap, interp, labels):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env.
    """
    states, _,real_traj_lengths,_ = collect_particles(env, policies, num_traj, traj_len)
    d_full, d_a1, d_a2 = compute_full_distributions(env, states, num_traj, real_traj_lengths)
    e_full, e_a1, e_a2 = - torch.sum(d_full*torch.log(d_full)), - torch.sum(d_a1*torch.log(d_a1)), - torch.sum(d_a2*torch.log(d_a2))
    plt.close()
    image_fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    log_p_1, log_p_2 = np.ma.log(d_a1), np.ma.log(d_a2)
    log_p_1_ravel, log_p_2_ravel = log_p_1.ravel(), log_p_2.ravel()

    # 
    axes[0].imshow(log_p_1.filled(np.min(log_p_1_ravel)), interpolation=interp, cmap=cmap)
    axes[0].set_title("Agent 1 Door Closed")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(labels[0])
    axes[0].set_ylabel(labels[1])
    # 
    axes[1].imshow(log_p_2.filled(np.min(log_p_2_ravel)), interpolation=interp, cmap=cmap)
    axes[1].set_title("Agent 2 Door Closed")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel(labels[0])
    axes[1].set_ylabel(labels[1])
    plt.tight_layout()

    return None, [e_a1.item(), e_a2.item()], image_fig


def collect_particles_parallel(env, policies, num_traj, traj_len, num_workers):
    # Collect particles using behavioral policy
    res = Parallel(n_jobs=num_workers)(
        delayed(collect_particles)(env, policies, int(num_traj/num_workers), traj_len)
        for _ in range(num_workers)
    )
    states, actions, real_traj_lengths, next_states = [np.vstack(x) for x in zip(*res)]
    # Return tensors so that downstream computations can be moved to any target device (#todo)
    states = torch.tensor(states, dtype=float_type)
    actions = torch.tensor(actions, dtype=float_type)
    next_states = torch.tensor(next_states, dtype=float_type)
    real_traj_lengths = torch.tensor(real_traj_lengths, dtype=int_type)
    return states, actions, real_traj_lengths, next_states


def collect_particles(env, policies, num_traj, traj_len):
    
    n_states = len(env.reset())
    n_actions = env.action_space.nvec.size
    raw_obs_dim = env.observation_space.nvec.size
    states = np.zeros((num_traj, traj_len + 1, n_states), dtype=np.float32)
    actions = np.zeros((num_traj, traj_len, n_actions), dtype=np.float32)
    real_traj_lengths = np.zeros((num_traj, 1), dtype=np.int32)

    for trajectory in range(num_traj):
        s = env.reset()
        for t in range(traj_len):
            states[trajectory, t] = s
            a = [policy.predict(s).item() for policy in policies]
            # ------ FULL-FEEDBACK --------
            actions[trajectory, t] = a

            ns, _, done = env.step(a)

            s = ns
            if done:
                break

        states[trajectory, t+1] = ns
        real_traj_lengths[trajectory] = t+1

    next_states = None
    for n_traj in range(num_traj):
        traj_real_len = real_traj_lengths[n_traj].item()
        traj_next_states = states[n_traj, 1:traj_real_len+1, :].reshape(-1, n_states)
        if next_states is None:
            next_states = traj_next_states
        else:
            next_states = np.concatenate([next_states, traj_next_states], axis=0)

    return states, actions, real_traj_lengths, next_states


def compute_importance_weights(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
    # Initialize to None for the first concat
    importance_weights = None
    importance_weights_a1 = None
    importance_weights_a2 = None
    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()

        traj_states = states[n_traj, :traj_length]
        traj_actions = actions[n_traj, :traj_length]
        traj_target_log_p = [target_policy.get_log_p(traj_states, traj_actions[:, id]) for id, target_policy in enumerate(target_policies)]
        traj_behavior_log_p = [behavioral_policy.get_log_p(traj_states, traj_actions[:,id]) for id, behavioral_policy in enumerate(behavioral_policies)]
        traj_particle_iw = torch.exp(torch.sum(traj_target_log_p[0] + traj_target_log_p[1] - traj_behavior_log_p[0] - traj_behavior_log_p[1], dim=0))
        traj_particle_iw_a1 = torch.exp(torch.sum(traj_target_log_p[0] - traj_behavior_log_p[0], dim=0))
        traj_particle_iw_a2 = torch.exp(torch.sum(traj_target_log_p[1] - traj_behavior_log_p[1], dim=0))
        
        if importance_weights is None:
            importance_weights = traj_particle_iw.unsqueeze(0)
            importance_weights_a1 = traj_particle_iw_a1.unsqueeze(0)
            importance_weights_a2 = traj_particle_iw_a2.unsqueeze(0)
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw.unsqueeze(0)], dim=0)
            importance_weights_a1 = torch.cat([importance_weights_a1, traj_particle_iw_a1.unsqueeze(0)], dim=0)
            importance_weights_a2 = torch.cat([importance_weights_a2, traj_particle_iw_a2.unsqueeze(0)], dim=0)

    # Normalize the weights
    importance_weights_norm = importance_weights / torch.sum(importance_weights)
    importance_weights_norm_a1 = importance_weights_a1 / torch.sum(importance_weights_a1)
    importance_weights_norm_a2 = importance_weights_a2 / torch.sum(importance_weights_a2)
    
    return importance_weights, importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2


def compute_full_distributions(env, states, num_traj, real_traj_lengths):
    dim_states = tuple(env.observation_space.nvec)
    a1_ind = [0,1]
    a2_ind = [2,3]
    dim_states_a1 = tuple(env.observation_space.nvec[a1_ind])
    dim_states_a2 = tuple(env.observation_space.nvec[a2_ind])
    states_counter = torch.zeros((num_traj,) + dim_states,  dtype=float_type)
    states_counter_a1 = torch.zeros((num_traj,) + dim_states_a1,  dtype=float_type)
    states_counter_a2 = torch.zeros((num_traj,) + dim_states_a2,  dtype=float_type)
    states = states.to(int_type)
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0]
        traj_states = states[n_traj, :traj_length]
        for state in traj_states:
            indices = (n_traj, state[0], state[1], state[2], state[3])
            indices_a1 = (n_traj, state[0], state[1])
            indices_a2 = (n_traj, state[2], state[3])
            states_counter[indices] +=1
            states_counter_a1[indices_a1] +=1
            states_counter_a2[indices_a2] +=1
        states_counter[n_traj, :] = (1 / traj_length) * states_counter[n_traj,:]
        states_counter_a1[n_traj, :] = (1 / traj_length) * states_counter_a1[n_traj,:]
        states_counter_a2[n_traj, :] = (1 / traj_length) * states_counter_a2[n_traj,:]
    states_counter = torch.clamp(states_counter, 1e-12, 1.0)
    states_counter_a1 = torch.clamp(states_counter_a1, 1e-12, 1.0)
    states_counter_a2 = torch.clamp(states_counter_a2, 1e-12, 1.0)
    return  (1 / num_traj) * torch.sum(states_counter, dim=0), (1 / num_traj) * torch.sum(states_counter_a1, dim=0), (1 / num_traj) * torch.sum(states_counter_a2, dim=0)


def compute_distributions(env, states, num_traj, real_traj_lengths):

    dim_states = tuple(env.observation_space.nvec)
    a1_ind = [0,1]
    a2_ind = [2,3]
    dim_states_a1 = tuple(env.observation_space.nvec[a1_ind])
    dim_states_a2 = tuple(env.observation_space.nvec[a2_ind])
    states_counter = torch.zeros((num_traj,) + dim_states,  dtype=float_type)
    states_counter_a1 = torch.zeros((num_traj,) + dim_states_a1,  dtype=float_type)
    states_counter_a2 = torch.zeros((num_traj,) + dim_states_a2,  dtype=float_type)
    states = states.to(int_type)
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0]
        traj_states = states[n_traj, :traj_length]
        for state in traj_states:
            indices = (n_traj, state[0], state[1], state[2], state[3])
            indices_a1 = (n_traj, state[0], state[1])
            indices_a2 = (n_traj, state[2], state[3])
            states_counter[indices] +=1
            states_counter_a1[indices_a1] +=1
            states_counter_a2[indices_a2] +=1
        states_counter[n_traj, :] = (1 / traj_length) * states_counter[n_traj,:]
        states_counter_a1[n_traj, :] = (1 / traj_length) * states_counter_a1[n_traj,:]
        states_counter_a2[n_traj, :] = (1 / traj_length) * states_counter_a2[n_traj,:]
    states_counter = torch.clamp(states_counter, 1e-12, 1.0)
    states_counter_a1 = torch.clamp(states_counter_a1, 1e-12, 1.0)
    states_counter_a2 = torch.clamp(states_counter_a2, 1e-12, 1.0)
    return states_counter, states_counter_a1, states_counter_a2

def compute_entropy(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
    _, importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2  = compute_importance_weights(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    # compute importance-weighted entropy
    distributions_per_traj, distributions_per_traj_a1, distributions_per_traj_a2 = compute_distributions(env, states, num_traj, real_traj_lengths)
    entropy_norm = - torch.sum(importance_weights_norm * torch.sum(distributions_per_traj*torch.log(distributions_per_traj), dim=tuple(range(1, len(distributions_per_traj.shape)))), dim=0)
    entropy_a1_norm = - torch.sum(importance_weights_norm_a1 * torch.sum(distributions_per_traj_a1*torch.log(distributions_per_traj_a1), dim=(1,2)), dim=0)
    entropy_a2_norm = - torch.sum(importance_weights_norm_a2 * torch.sum(distributions_per_traj_a2*torch.log(distributions_per_traj_a2), dim=(1,2)), dim=0)

    return entropy_norm, entropy_a1_norm, entropy_a2_norm


def compute_mutual_information(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
    _, importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2  = compute_importance_weights(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    # compute importance-weighted entropy
    distributions_per_traj, distributions_per_traj_a1, distributions_per_traj_a2 = compute_distributions(env, states, num_traj, real_traj_lengths)
    mi = - torch.sum(importance_weights_norm * torch.sum(distributions_per_traj*(torch.log(distributions_per_traj) - torch.log(distributions_per_traj_a1) - torch.log(distributions_per_traj_a2)), dim=tuple(range(1, len(distributions_per_traj.shape)))), dim=0)
    return mi, mi



def compute_kl(behavioral_policies, target_policies, states):
    # numpy_states = torch.from_numpy(states).type(float_type)
    numeric_error = False
    kls = []
    # Compute KL divergence between behavioral and target policy
    for behavioral_policy, target_policy in zip(behavioral_policies, target_policies):
        p0, _, log_p0 = behavioral_policy.forward(states)
        p1, _, log_p1 = target_policy.forward(states)
        kl = torch.sum(p0*(torch.log(p0) -torch.log(p1)), dim=(0,1)).mean()
        kls.append(kl)
        numeric_error = torch.isinf(kl) or torch.isnan(kl) or numeric_error
    # Minimum KL is zero
    kls = [torch.max(torch.tensor(0.0), kl) for kl in kls]
    return kls, numeric_error


def log_epoch_statistics(writer, log_file, csv_file_1, csv_file_2, epoch,
                         loss, entropy, num_off_iters, execution_time, full_entropy,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss A1", loss[0], global_step=epoch)
    writer.add_scalar("Loss A2", loss[1], global_step=epoch)
    writer.add_scalar("Est Entropy", entropy[0], global_step=epoch)
    writer.add_scalar("Est Entropy A1", entropy[1], global_step=epoch)
    writer.add_scalar("Est Entropy A2", entropy[2], global_step=epoch)
    writer.add_scalar("Est MI", entropy[3], global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)
    if full_entropy is not None:
        writer.add_scalar(f"Full Entropy:", full_entropy, global_step=epoch)

    if heatmap_image is not None:
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["Est Entropy", fancy_float(entropy[0])],
        ["Est Entropy A1", fancy_float(entropy[1])],
        ["Est Entropy A2", fancy_float(entropy[2])],
        ["Est MI", fancy_float(entropy[3])],
        ["Off-policy iters", num_off_iters]
    ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch}, {loss[0]}, {loss[1]}, {entropy[0]}, {entropy[1]}, {entropy[2]}, {entropy[3]}, {full_entropy}, {num_off_iters}, {execution_time}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy[0]}, {heatmap_entropy[1]}\n")
        csv_file_2.flush()
        table.extend([
            ["True Entropy A1", fancy_float(heatmap_entropy[0])],
            ["True Entropy A2", fancy_float(heatmap_entropy[1])]
        ])

    # Log to stdout and log file
    log_file.write(fancy_grid)
    log_file.flush()
    print(fancy_grid)


def log_off_iter_statistics(writer, csv_file_3, epoch, global_off_iter,
                            num_off_iter, entropy, kls, lr):
    # Log to csv file 3
    csv_file_3.write(f"{epoch},{num_off_iter}, {entropy[0]}, {entropy[1]}, {kls[0]}, {kls[1]}, {lr}\n")
    # Also log to tensorboard
    writer.add_scalar("Off policy iter Loss A1", entropy[0], global_step=global_off_iter)
    writer.add_scalar("Off policy iter Loss A2", entropy[1], global_step=global_off_iter)
        
    csv_file_3.flush()
    writer.add_scalar("Off policy iter KL Agent 1", kls[0], global_step=global_off_iter)
    writer.add_scalar("Off policy iter KL Agent 2", kls[1], global_step=global_off_iter)


def policy_update(env, thresholds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, traj_len, update_algo):
    # Maximize entropy
    entropy, entropy_a1, entropy_a2 =  compute_entropy(env, behavioral_policies, target_policies, states, actions, num_traj, traj_len)
    mi_a1, mi_a2 = compute_mutual_information(env, behavioral_policies, target_policies, states, actions, num_traj, traj_len)
    if update_algo == "Centralized":
        loss = [- entropy, -entropy]
        numeric_error = torch.isinf(loss) or torch.isnan(loss)
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].zero_grad()
            loss[id].backward()
            optimizers[id].step()
    elif update_algo == "Decentralized":
        loss = [- entropy_a1, - entropy_a2]
        numeric_error = torch.isinf(loss[0]) or torch.isinf(loss[1]) or torch.isnan(loss[0]) or torch.isnan(loss[1]) 
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].zero_grad()
            loss[id].backward()
            optimizers[id].step()
    elif update_algo == "Decentralized MI":
        loss = [- entropy_a1 + mi_a1, - entropy_a2 + mi_a2]
        numeric_error = torch.isinf(loss[0]) or torch.isinf(loss[1]) or torch.isnan(loss[0]) or torch.isnan(loss[1]) 
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].zero_grad()
            loss[id].backward()
            optimizers[id].step()
    else:
        raise NotImplementedError

    return loss, numeric_error


def mamepol(env, env_name, state_filter, create_policy, k, kl_threshold, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          learning_rate, num_traj, traj_len, num_epochs, optimizer,
          full_entropy_traj_scale, full_entropy_k,
          heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
          heatmap_cmap, heatmap_labels, heatmap_interp,
          seed, out_path, num_workers, update_algo):

    # Seed everything
    if seed is not None:
        # Seed everything
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed)

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policies = []
    target_policies = []
    for agent in range(env.n_agents):
        behavioral_policies.concatenate([create_policy()])
        target_policies.concatenate([create_policy()])
        last_valid_target_policies.concatenate([create_policy()])
        target_policies[agent].load_state_dict(behavioral_policies[agent].state_dict())
        last_valid_target_policies[agent].load_state_dict(behavioral_policies[agent].state_dict())

    # Set optimizer
    optimizers = []
    if optimizer == 'rmsprop':
        for agent in range(env.n_agents):
            optimizers.concatenate([torch.optim.RMSprop(target_policies[agent].parameters(), lr=learning_rate)])
    elif optimizer == 'adam':
        for agent in range(env.n_agents):
            optimizers.concatenate([torch.optim.Adam(target_policies[agent].parameters(), lr=learning_rate)])
    else:
        raise NotImplementedError
    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")
    csv_file_1 = open(os.path.join(out_path, f"{env_name}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss A1', 'loss A2', 'Est entropy', 'Est entropy A1', 'Est entropy A2', 'Est MI', 'full_entropy', 'num_off_iters','execution_time']))
    csv_file_1.write("\n")

    
    if heatmap_discretizer is not None:
        csv_file_2 = open(os.path.join(out_path, f"{env_name}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'average_entropy']))
        csv_file_2.write("\n")
    else:
        csv_file_2 = None

    csv_file_3 = open(os.path.join(out_path, f"{env_name}_off_policy_iter.csv"), "w")
    csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'Entropy A1', 'Entropy A2', 'kl A1', 'kl A2', 'learning_rate']))
    csv_file_3.write("\n")

    # Fixed constants
    ns = len(state_filter) if (state_filter is not None) else env.num_features
    B = np.log(k) - scipy.special.digamma(k)
    full_B = np.log(full_entropy_k) - scipy.special.digamma(full_entropy_k)
    G = scipy.special.gamma(ns/2 + 1)
    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    t0 = time.time()

    # Entropy Computation
    states, actions, real_traj_lengths, next_states = collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers)

    with torch.no_grad():
        entropy, entropy_a1, entropy_a2 = compute_entropy(env, behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths)
        mi = compute_mu(env, behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths)
    execution_time = time.time() - t0
    entropy = entropy.numpy()
    if update_algo == "Centralized":
        loss = [- entropy, -entropy]
    elif update_algo == "Decentralized":
        loss = [- entropy_a1.numpy(), -entropy_a2.numpy()]
    else:
        raise NotImplementedError

    # Heatmap
    if heatmap_discretizer is not None:
        _, heatmap_entropies, heatmap_image = get_heatmap(env, behavioral_policies, num_traj, traj_len, heatmap_cmap, heatmap_interp, heatmap_labels)
    else:
        heatmap_entropies = None
        heatmap_image = None
    
    # Save initial policy
    for agent in range(env.n_agents):
        torch.save(behavioral_policies[agent].state_dict(), os.path.join(out_path, f"{epoch}-policy-{agent}"))

    # Log statistics for the initial policy
    log_epoch_statistics(
            writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=epoch,
            loss=loss,
            entropy=[entropy, entropy_a1, entropy_a2, mi],
            execution_time=execution_time,
            num_off_iters=0,
            full_entropy=None,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropies,
            backtrack_iters=None,
            backtrack_lr=None
        )

    # Main Loop
    global_num_off_iters = 0

    if use_backtracking:
        original_lr = learning_rate

    while epoch < num_epochs:
        t0 = time.time()

        # Off policy optimization
        kl_threshold_reacheds = [False, False]

        for agent in range(env.n_agents):
            last_valid_target_policies[agent].load_state_dict(behavioral_policies[agent].state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, real_traj_lengths, next_states = collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers)

        if use_backtracking:
            learning_rate = original_lr
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            backtrack_iter = 1
        else:
            backtrack_iter = None

        while not all(kl_threshold_reacheds):
            # Optimize policy
            loss, numeric_error = policy_update(env, kl_threshold_reacheds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths, update_type = update_algo)
            entropy= [- loss[0].detach().numpy(), -loss[1].detach().numpy()]
            with torch.no_grad():
                kls, kl_numeric_error = compute_kl(behavioral_policies, target_policies, states)

            kls = [kl.numpy() for kl in kls]
            if not numeric_error and not kl_numeric_error and any([kl <= kl_threshold for kl in kls]):
                # Valid update
                for agent in range(env.n_agents):
                    last_valid_target_policies[agent].load_state_dict(target_policies[agent].state_dict())
            
                num_off_iters += 1
                global_num_off_iters += 1

                # Log statistics for this off policy iteration
                log_off_iter_statistics(writer, csv_file_3, epoch, global_num_off_iters, num_off_iters - 1, entropy, kls, learning_rate)

                # Update for which agent the process should go on
                kl_threshold_reacheds = [kl >= kl_threshold for kl in kls]
            else:
                if use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == max_backtrack_try:
                        for agent in range(env.n_agents):
                            target_policies[agent].load_state_dict(last_valid_target_policies[agent].state_dict())

                        learning_rate = original_lr / (backtrack_coeff ** backtrack_iter)
                        for id in [i for i, v in enumerate(kl_threshold_reacheds) if not v]:
                            for param_group in optimizers[id].param_groups:
                                param_group['lr'] = learning_rate

                        backtrack_iter += 1
                        continue

                # Do not accept the update, set exit condition to end the epoch
                kl_threshold_reacheds = [True, True]

            if use_backtracking and backtrack_iter > 1:
                # Just perform at most 1 step using backtracking
                kl_threshold_reacheds = [True, True]

            if num_off_iters == max_off_iters:
                # Set exit condition also if the maximum number
                # of off policy opt iterations has been reached
                kl_threshold_reacheds = [True, True]

            if all(kl_threshold_reacheds):
                # Compute entropy of new policy
                with torch.no_grad():
                    entropy, entropy_a1, entropy_a2 = compute_entropy(env, last_valid_target_policies, last_valid_target_policies, states, actions, num_traj, real_traj_lengths)

                if torch.isnan(entropy) or torch.isinf(entropy):
                    print("Aborting because final entropy is nan or inf...")
                    exit()
                else:
                    # End of epoch, prepare statistics to log
                    epoch += 1

                    # Update behavioral policy
                    for agent in range(env.n_agents):
                        behavioral_policies[agent].load_state_dict(last_valid_target_policies[agent].state_dict())
                        target_policies[agent].load_state_dict(last_valid_target_policies[agent].state_dict())

                    if update_algo == "Centralized":
                        loss = [- entropy.numpy(), -entropy.numpy()]
                    elif update_algo == "Decentralized":
                        loss = [- entropy_a1.numpy(), -entropy_a2.numpy()]
                    else:
                        raise NotImplementedError
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None:
                            _, heatmap_entropies, heatmap_image = \
                                get_heatmap(env, behavioral_policies, num_traj, traj_len, heatmap_cmap, heatmap_interp, heatmap_labels)
                        else:
                            heatmap_entropies = None
                            heatmap_image = None

                        # Save policy
                        for agent in range(env.n_agents):
                            torch.save(behavioral_policies[agent].state_dict(), os.path.join(out_path, f"{epoch}-policy-{agent}"))

                    else:
                        heatmap_entropies = None
                        heatmap_image = None

                    # Log statistics for this epoch
                    log_epoch_statistics(
                        writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                        epoch=epoch,
                        loss=loss,
                        entropy=[entropy, entropy_a1, entropy_a2, mi],
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        full_entropy=None,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropies,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=learning_rate
                    )

    return behavioral_policies