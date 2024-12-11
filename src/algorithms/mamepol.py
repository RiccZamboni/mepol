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


def get_heatmap(env, policies, num_traj, real_traj_lengths, counter_dimension, traj_len, num_workers, num_episodes, num_steps, cmap, interp, labels):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env.
    """
    states, _, real_traj_lengths, _ = collect_particles(env, policies, num_traj, traj_len, num_workers)
    d_0, d_1 = compute_per_agent_distribution(states, num_traj, real_traj_lengths, counter_dimension)
    e_0, e_1 = scipy.stats.entropy(d_0.ravel()), scipy.stats.entropy(d_1.ravel())
    plt.close()
    image_fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    if len(d_0.shape) == 2 and len(d_1.shape) == 2:
        log_p_0, log_p_1 = np.ma.log(d_0), np.ma.log(d_1)
        log_p_0_ravel, log_p_1_ravel = log_p_0.ravel(), log_p_1.ravel()
        min_log_p_0_ravel, min_log_p_1_ravel = np.min(log_p_0_ravel),  np.min(log_p_1_ravel)
        second_min_log_p_0_ravel, second_min_log_p_1_ravel = np.min(log_p_0_ravel[log_p_0_ravel != min_log_p_0_ravel]), np.min(log_p_1_ravel[log_p_1_ravel != min_log_p_1_ravel])
        log_p_0_ravel[np.argmin(log_p_0_ravel)] = second_min_log_p_0_ravel
        log_p_1_ravel[np.argmin(log_p_1_ravel)] = second_min_log_p_1_ravel

        # First subplot
        axes[0].imshow(log_p_0.filled(min_log_p_0_ravel), interpolation=interp, cmap=cmap)
        axes[0].set_title("Agent 1")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])
        # Second subplot
        axes[1].imshow(log_p_1.filled(min_log_p_1_ravel), interpolation=interp, cmap=cmap)
        axes[1].set_title("Agent 2")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_xlabel(labels[0])
        axes[1].set_ylabel(labels[1])

        
    else:
        raise NotImplementedError
    plt.tight_layout()
    dist = [d_0,d_1]
    ent = [e_0,e_1]
    return dist, ent, image_fig


def collect_particles(env, policies, num_traj, traj_len):
    """
    Collects num_traj * traj_len samples by running policy in the env.
    """
    # ------ FULL-FEEDBACK --------
    n_states = reduce(lambda x, y: x * y, env.observation_space.nvec)
    n_actions = reduce(lambda x, y: x * y, env.action_space.nvec)
    raw_obs_dim = env.observation_space.nvec.size
    # ------ FULL-FEEDBACK --------
    states = np.zeros((num_traj, traj_len + 1, n_states), dtype=np.float32)
    actions = np.zeros((num_traj, traj_len, n_actions), dtype=np.float32)
    real_traj_lengths = np.zeros((num_traj, 1), dtype=np.int32)

    for trajectory in range(num_traj):
        s_raw = env.reset()
        # ------ FULL-FEEDBACK --------
        s, s_p = obs_to_int_pi(s_raw, base=env.grid_size, raw_dim=raw_obs_dim)
        # ------ FULL-FEEDBACK --------
        for t in range(traj_len):
            states[trajectory, t] = s_raw
            # ------ FULL-FEEDBACK --------
            a = [policy.predict(s).numpy() for policy in policies]
            # ------ FULL-FEEDBACK --------
            actions[trajectory, t] = a

            ns, _, done, _ = env.step(a)

            s = ns

            if done:
                break

        states[trajectory, t+1] = s
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

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()

        traj_states = states[n_traj, :traj_length]
        traj_actions = actions[n_traj, :traj_length]
        # ------ FULL-FEEDBACK --------
        traj_target_log_p = [target_policy.get_log_p(traj_states, traj_actions) for target_policy in target_policies]
        traj_behavior_log_p = [behavioral_policy.get_log_p(traj_states, traj_actions) for behavioral_policy in behavioral_policies]
        traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p[0] + traj_target_log_p[1] - traj_behavior_log_p[0] - traj_behavior_log_p[1], dim=0))
        # ------ FULL-FEEDBACK --------
        if importance_weights is None:
            importance_weights = traj_particle_iw
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)

    # Normalize the weights
    importance_weights /= torch.sum(importance_weights)
    return importance_weights

def compute_per_agent_distribution(states, num_traj, real_traj_lengths, counter_dimension):
    # ------ FULL-FEEDBACK --------
    # Compute the frequencies
    states_counter_0 = np.zeros((counter_dimension, counter_dimension), dtype=np.int32)
    states_counter_1 = np.zeros((counter_dimension, counter_dimension), dtype=np.int32)
    states_traj_counter_0 = np.zeros((num_traj, counter_dimension, counter_dimension), dtype=np.int32)
    states_traj_counter_1 = np.zeros((num_traj, counter_dimension, counter_dimension), dtype=np.int32)
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()
        traj_states = states[n_traj, :traj_length]
        for state in traj_states:
            states_traj_counter_0[n_traj, state[1], state[0]] +=1
            states_traj_counter_1[n_traj, state[1], state[0]] +=1
        states_traj_counter_0[n_traj, :] = (1 / traj_length) * states_traj_counter_0[n_traj,:]
        states_traj_counter_1[n_traj, :] = (1 / traj_length) * states_traj_counter_1[n_traj,:]
    # ------ FULL-FEEDBACK --------
    states_counter_0 = torch.mean(states_traj_counter_0, dim=0)
    states_counter_1 = torch.mean(states_traj_counter_1, dim=0)
    return states_counter_0, states_counter_1


def compute_distributions(states, num_traj, real_traj_lengths, counter_dimension):
    # ------ FULL-FEEDBACK --------
    # Compute the frequencies
    states_counter = np.zeros((num_traj, counter_dimension, counter_dimension, counter_dimension, counter_dimension), dtype=np.int32)
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()
        traj_states = states[n_traj, :traj_length]
        for state in traj_states:
            states_counter[n_traj, state[1], state[0], state[3], state[2]] +=1
        states_counter[n_traj, :] = (1 / traj_length) * states_counter[n_traj,:]
    # ------ FULL-FEEDBACK --------
    return states_counter

def compute_entropy(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths, counter_dimension):
    importance_weights = compute_importance_weights(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    # ------ FULL-FEEDBACK --------
    # compute importance-weighted entropy
    distributions_per_traj = compute_distributions(states, num_traj, real_traj_lengths, counter_dimension)
    entropy = - torch.sum(importance_weights * torch.sum(distributions_per_traj*torch.log(distributions_per_traj), dim=(1,2,3,4)), dim=0)
    # ------ FULL-FEEDBACK --------
    return entropy


def compute_kl(behavioral_policies, target_policies, states):
    numpy_states = torch.from_numpy(states).type(float_type)
    numeric_error = False
    kls = []
    # Compute KL divergence between behavioral and target policy
    for behavioral_policy, target_policy in zip(behavioral_policies, target_policies):
        p0 = behavioral_policy(numpy_states).detach()
        p1 = target_policy(numpy_states)
        kl = (p0*torch.log(p0/p1)).sum(dim=1).mean()
        kls.append(kl)
        numeric_error = torch.isinf(kl) or torch.isnan(kl) or numeric_error
    # Minimum KL is zero
    kls = torch.max(torch.tensor(0.0), kls)
    return kls, numeric_error


def log_epoch_statistics(writer, log_file, csv_file_1, csv_file_2, epoch,
                         loss, entropy, num_off_iters, execution_time, full_entropy,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss", loss, global_step=epoch)
    writer.add_scalar("Entropy", entropy, global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)
    if full_entropy is not None:
        writer.add_scalar(f"Full Entropy:", full_entropy, global_step=epoch)

    if heatmap_image is not None:
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)
        writer.add_scalar(f"Discrete entropy", heatmap_entropy, global_step=epoch)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["Entropy", fancy_float(entropy)],
        ["Off-policy iters", num_off_iters]
    ])

    if heatmap_image is not None:
        table.extend([
            ["Heatmap entropy", fancy_float(heatmap_entropy)]
        ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch},{loss},{entropy},{full_entropy},{num_off_iters},{execution_time}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy}\n")
        csv_file_2.flush()

    # Log to stdout and log file
    log_file.write(fancy_grid)
    log_file.flush()
    print(fancy_grid)


def log_off_iter_statistics(writer, csv_file_3, epoch, global_off_iter,
                            num_off_iter, entropy, kl, lr):
    # Log to csv file 3
    csv_file_3.write(f"{epoch},{num_off_iter},{entropy},{kl},{lr}\n")
    csv_file_3.flush()

    # Also log to tensorboard
    writer.add_scalar("Off policy iter Entropy", entropy, global_step=global_off_iter)
    writer.add_scalar("Off policy iter KL", kl, global_step=global_off_iter)


def policy_update(thresholds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, traj_len):
    # Maximize entropy
    loss = compute_entropy(behavioral_policies, target_policies, states, actions, num_traj, traj_len)
    numeric_error = torch.isinf(loss) or torch.isnan(loss)
    for id in [i for i, v in enumerate(thresholds) if not v]:
        optimizers[id].zero_grad()
        loss.backward()
        optimizers[id].step()

    return loss, numeric_error


def mamepol(env, env_name, state_filter, create_policy, k, kl_threshold, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          learning_rate, num_traj, traj_len, num_epochs, optimizer,
          full_entropy_traj_scale, full_entropy_k,
          heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
          heatmap_cmap, heatmap_labels, heatmap_interp,
          seed, out_path, num_workers):

    # Seed everything
    if seed is not None:
        # Seed everything
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    # ------ FULL-FEEDBACK --------
    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policies = [create_policy(is_behavioral=True), create_policy(is_behavioral=True)]
    target_policies = [create_policy(), create_policy()]
    last_valid_target_policies = [create_policy(), create_policy()]
    target_policies[0].load_state_dict(behavioral_policies[0].state_dict())
    target_policies[1].load_state_dict(behavioral_policies[1].state_dict())
    last_valid_target_policies[0].load_state_dict(behavioral_policies[0].state_dict())
    last_valid_target_policies[1].load_state_dict(behavioral_policies[1].state_dict())
    # ------ FULL-FEEDBACK --------

    # Set optimizer
    if optimizer == 'rmsprop':
        optimizers = [torch.optim.RMSprop(target_policies[0].parameters(), lr=learning_rate),
                      torch.optim.RMSprop(target_policies[1].parameters(), lr=learning_rate)]
    elif optimizer == 'adam':
        optimizers = [torch.optim.Adam(target_policies[0].parameters(), lr=learning_rate),
                      torch.optim.Adam(target_policies[1].parameters(), lr=learning_rate)]
    else:
        raise NotImplementedError
    # ------ FULL-FEEDBACK --------
    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")

    csv_file_1 = open(os.path.join(out_path, f"{env_name}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss', 'entropy', 'full_entropy', 'num_off_iters','execution_time']))
    csv_file_1.write("\n")

    if heatmap_discretizer is not None:
        csv_file_2 = open(os.path.join(out_path, f"{env_name}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'average_entropy']))
        csv_file_2.write("\n")
    else:
        csv_file_2 = None

    csv_file_3 = open(os.path.join(out_path, f"{env_name}_off_policy_iter.csv"), "w")
    csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'entropy', 'kl', 'learning_rate']))
    csv_file_3.write("\n")

    # Fixed constants
    ns = len(state_filter) if (state_filter is not None) else env.num_features
    B = np.log(k) - scipy.special.digamma(k)
    full_B = np.log(full_entropy_k) - scipy.special.digamma(full_entropy_k)
    G = scipy.special.gamma(ns/2 + 1)
    # ------ FULL-FEEDBACK --------
    # MPE Environments Constants
    # ------ FULL-FEEDBACK --------
    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    t0 = time.time()

    # Entropy Computation
    states, actions, real_traj_lengths, next_states = collect_particles(env, behavioral_policies, num_traj, traj_len, num_workers)

    with torch.no_grad():
        entropy = compute_entropy(behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths, env.grid_size)
    execution_time = time.time() - t0
    full_entropy = full_entropy.numpy()
    entropy = entropy.numpy()
    loss = - entropy

    # Heatmap
    # ------ FULL-FEEDBACK --------
    if heatmap_discretizer is not None:
        _, heatmap_entropies, heatmap_image = get_heatmap(env, states, num_traj, real_traj_lengths, env.grid_size, heatmap_episodes, heatmap_num_steps, heatmap_cmap, heatmap_interp, heatmap_labels)
    else:
        heatmap_entropy = None
        heatmap_image = None
    
    # Save initial policy
    torch.save(behavioral_policies[0].state_dict(), os.path.join(out_path, f"{epoch}-policy-0"))
    torch.save(behavioral_policies[1].state_dict(), os.path.join(out_path, f"{epoch}-policy-1"))
    # ------ FULL-FEEDBACK --------

    # Log statistics for the initial policy
    log_epoch_statistics(
            writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=epoch,
            loss=loss,
            entropy=entropy,
            execution_time=execution_time,
            num_off_iters=0,
            full_entropy=None,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
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
        
        last_valid_target_policies[0].load_state_dict(behavioral_policies[0].state_dict())
        last_valid_target_policies[1].load_state_dict(behavioral_policies[1].state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, real_traj_lengths, next_states = collect_particles(env, behavioral_policies, num_traj, traj_len, num_workers)

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
            
            loss, numeric_error = policy_update(kl_threshold_reacheds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
            entropy = - loss.detach().numpy()

            with torch.no_grad():
                kls, kl_numeric_error = compute_kl(behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)

            kls = kls.numpy()
            if not numeric_error and not kl_numeric_error and any(kls <= kl_threshold):
                # Valid update
                last_valid_target_policies[0].load_state_dict(target_policies[0].state_dict())
                last_valid_target_policies[1].load_state_dict(target_policies[1].state_dict())
            
                num_off_iters += 1
                global_num_off_iters += 1

                # Log statistics for this off policy iteration
                log_off_iter_statistics(writer, csv_file_3, epoch, global_num_off_iters, num_off_iters - 1, entropy, kls, learning_rate)

                # Update for which agent the process should go on
                kl_threshold_reacheds = kls >= kl_threshold
            else:
                if use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == max_backtrack_try:
                        target_policies[0].load_state_dict(last_valid_target_policies[0].state_dict())
                        target_policies[1].load_state_dict(last_valid_target_policies[1].state_dict())

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
                    entropy = compute_entropy(last_valid_target_policies, last_valid_target_policies, states, actions, num_traj, real_traj_lengths, env.grid_size)

                if torch.isnan(entropy) or torch.isinf(entropy):
                    print("Aborting because final entropy is nan or inf...")
                    print("There is most likely a problem in knn aliasing. Use a higher k.")
                    exit()
                else:
                    # End of epoch, prepare statistics to log
                    epoch += 1

                    # Update behavioral policy
                    behavioral_policies[0].load_state_dict(last_valid_target_policies[0].state_dict())
                    behavioral_policies[1].load_state_dict(last_valid_target_policies[1].state_dict())
                    target_policies[0].load_state_dict(last_valid_target_policies[0].state_dict())
                    target_policies[1].load_state_dict(last_valid_target_policies[1].state_dict())

                    loss = - entropy.numpy()
                    entropy = entropy.numpy()
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None:
                            _, heatmap_entropies, heatmap_image = \
                                get_heatmap(env, states, num_traj, real_traj_lengths, env.grid_size, heatmap_episodes, heatmap_num_steps, heatmap_cmap, heatmap_interp, heatmap_labels)
                        else:
                            heatmap_entropy = None
                            heatmap_image = None
                        full_entropy = full_entropy.numpy()

                        # Save policy
                        torch.save(behavioral_policies[0].state_dict(), os.path.join(out_path, f"{epoch}-policy-0"))
                        torch.save(behavioral_policies[1].state_dict(), os.path.join(out_path, f"{epoch}-policy-1"))

                    else:
                        heatmap_entropy = None
                        heatmap_image = None

                    # Log statistics for this epoch
                    log_epoch_statistics(
                        writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                        epoch=epoch,
                        loss=loss,
                        entropy=entropy,
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        full_entropy=None,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropy,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=learning_rate
                    )

    return behavioral_policies