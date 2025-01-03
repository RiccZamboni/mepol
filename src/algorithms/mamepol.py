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

# TODO ADD CONTINUOUS CASES
def get_heatmap(env, policies, heatmap_discretizer, num_traj, traj_len, cmap, interp, labels):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env.
    """
    states, _,real_traj_lengths,_ = collect_particles(env, policies, num_traj, traj_len)
    states = torch.tensor(states, dtype=float_type)
    d_full, d_a1, d_a2 = compute_full_distributions(env, states, num_traj, real_traj_lengths)
    e_full, e_a1, e_a2 = - torch.sum(d_full*torch.log(d_full)), - torch.sum(d_a1*torch.log(d_a1)), - torch.sum(d_a2*torch.log(d_a2))
    plt.close()
    image_fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    log_p_1, log_p_2 = np.ma.log(d_a1), np.ma.log(d_a2)
    log_p_1_ravel, log_p_2_ravel = log_p_1.ravel(), log_p_2.ravel()
    axes[0].imshow(log_p_1.filled(np.min(log_p_1_ravel)), interpolation=interp, cmap=cmap)
    axes[0].set_title("Agent 1")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(labels[0])
    axes[0].set_ylabel(labels[1])
    # 
    axes[1].imshow(log_p_2.filled(np.min(log_p_2_ravel)), interpolation=interp, cmap=cmap)
    axes[1].set_title("Agent 2")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel(labels[0])
    axes[1].set_ylabel(labels[1])
    plt.tight_layout()

    return None, [e_full.item(), e_a1.item(), e_a2.item()], image_fig


def collect_particles_parallel(env, policies, num_traj, traj_len, num_workers, state_filter, k):
    assert num_traj % num_workers == 0, "Please provide a number of trajectories " \
                                        "that can be equally split among workers"
    # Collect particles using behavioral policy
    res = Parallel(n_jobs=num_workers)(
        delayed(collect_particles)(env, policies, int(num_traj/num_workers), traj_len, state_filter)
        for _ in range(num_workers)
    )
    states, actions, real_traj_lengths, next_states = [np.vstack(x) for x in zip(*res)]
    distances = []
    indices = []
    # Fit knn for the batch of collected particles
    if not env.discrete:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto', n_jobs=num_workers)
        nbrs.fit(next_states)
        dist, ind = nbrs.kneighbors(next_states)
        distances.append(dist)
        indices.append(ind)
        nbrs.fit(next_states[:, env.state_indeces[0]])
        dist, ind = nbrs.kneighbors(next_states[:, env.state_indeces[0]])
        distances.append(dist)
        indices.append(ind)
        nbrs.fit(next_states[:, env.state_indeces[1]])
        dist, ind = nbrs.kneighbors(next_states[:, env.state_indeces[1]])
        distances.append(dist)
        indices.append(ind)
    # Return tensors so that downstream computations can be moved to any target device (#todo)
    states = torch.tensor(states, dtype=float_type)
    actions = torch.tensor(actions, dtype=float_type)
    next_states = torch.tensor(next_states, dtype=float_type)
    real_traj_lengths = torch.tensor(real_traj_lengths, dtype=int_type)
    return states, actions, real_traj_lengths, next_states, distances, indices


def collect_particles(env, policies, num_traj, traj_len, state_filter = None):
    
    n_states = len(env.reset())
    action_dim = env.action_dim
    states = np.zeros((num_traj, traj_len + 1, n_states), dtype=np.float32)
    actions = np.zeros((num_traj, traj_len, action_dim), dtype=np.float32)
    real_traj_lengths = np.zeros((num_traj, 1), dtype=np.int32)
    for trajectory in range(num_traj):
        s = env.reset()
        for t in range(traj_len):
            states[trajectory, t] = s
            a = []
            for id, policy in enumerate(policies):
                if action_dim > 1:
                    a = a + policy.predict(s).tolist() if not policy.policy_decentralized else a + policy.predict(s[env.state_indeces[id]]).tolist()
                else:
                    a = a + [policy.predict(s).item()] if not policy.policy_decentralized else a + [policy.predict(s[env.state_indeces[id]]).item()]
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
        traj_next_states = states[n_traj, 1:traj_real_len+1, :]
        if next_states is None:
            next_states = traj_next_states
        else:
            next_states = np.concatenate([next_states, traj_next_states], axis = 0)

    return states, actions, real_traj_lengths, next_states


def compute_importance_weights(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
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
        traj_target_log_p = []
        traj_behavior_log_p = []
        for id, target_policy in enumerate(target_policies):
            traj_target_log_p = traj_target_log_p + [target_policy.get_log_p(traj_states, traj_actions[:, env.action_indeces[id]])] if not target_policy.policy_decentralized else traj_target_log_p + [target_policy.get_log_p(traj_states[:, env.state_indeces[id]], traj_actions[:, env.action_indeces[id]])]
            assert True
        for id, behavioral_policy in enumerate(behavioral_policies):
            traj_behavior_log_p = traj_behavior_log_p + [behavioral_policy.get_log_p(traj_states, traj_actions[:, env.action_indeces[id]])] if not behavioral_policy.policy_decentralized else traj_behavior_log_p + [behavioral_policy.get_log_p(traj_states[:,env.state_indeces[id]], traj_actions[:, env.action_indeces[id]])]
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

def compute_importance_weights_knn(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
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
        traj_target_log_p = []
        traj_behavior_log_p = []
        for id, target_policy in enumerate(target_policies):
            traj_target_log_p = traj_target_log_p + [target_policy.get_log_p(traj_states, traj_actions[:, env.action_indeces[id]])] if not target_policy.policy_decentralized else traj_target_log_p + [target_policy.get_log_p(traj_states[:,env.state_indeces[id]], traj_actions[:, env.action_indeces[id]])]
        for id, behavioral_policy in enumerate(behavioral_policies):
            traj_behavior_log_p = traj_behavior_log_p + [behavioral_policy.get_log_p(traj_states, traj_actions[:, env.action_indeces[id]])] if not behavioral_policy.policy_decentralized else traj_behavior_log_p + [behavioral_policy.get_log_p(traj_states[:,env.state_indeces[id]], traj_actions[:, env.action_indeces[id]])]

        traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p[0] + traj_target_log_p[1] - traj_behavior_log_p[0] - traj_behavior_log_p[1], dim=0))
        traj_particle_a1 = torch.exp(torch.cumsum(traj_target_log_p[0] - traj_behavior_log_p[0], dim=0))
        traj_particle_a2 = torch.exp(torch.cumsum(traj_target_log_p[1] - traj_behavior_log_p[1], dim=0))

        if importance_weights is None:
            importance_weights = traj_particle_iw
            importance_weights_a1 = traj_particle_a1
            importance_weights_a2 = traj_particle_a2
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)
            importance_weights_a1 = torch.cat([importance_weights_a1, traj_particle_a1], dim=0)
            importance_weights_a2 = torch.cat([importance_weights_a2, traj_particle_a2], dim=0)

    # Normalize the weights
    importance_weights_norm = importance_weights / torch.sum(importance_weights)
    importance_weights_norm_a1 = importance_weights_a1 / torch.sum(importance_weights_a1)
    importance_weights_norm_a2 = importance_weights_a2 / torch.sum(importance_weights_a2)
    return importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2

# TODO FIX FOR CONTINUOUS
def compute_full_distributions(env, states, num_traj, real_traj_lengths):
    dim_states = tuple(env.observation_space.nvec[:-1])
    a12_ind = env.distribution_indices[0]
    a1_ind = env.distribution_indices[1]
    a2_ind = env.distribution_indices[2]
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
            indices = (n_traj,) + tuple(state[a12_ind].tolist())
            indices_a1 = (n_traj,) + tuple(state[a1_ind].tolist())
            indices_a2 = (n_traj,) + tuple(state[a2_ind].tolist())
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
    a12_ind = env.distribution_indices[0]
    a1_ind = env.distribution_indices[1]
    a2_ind = env.distribution_indices[2]
    dim_states = tuple(env.observation_space.nvec[a12_ind])
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
            indices = (n_traj,) + tuple(state[a12_ind].tolist())
            indices_a1 = (n_traj,) + tuple(state[a1_ind].tolist())
            indices_a2 = (n_traj,) + tuple(state[a2_ind].tolist())
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
    importance_weights, importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2  = compute_importance_weights(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    # compute importance-weighted entropy
    distributions_per_traj, distributions_per_traj_a1, distributions_per_traj_a2 = compute_distributions(env, states, num_traj, real_traj_lengths)
    entropy_norm = - torch.sum(importance_weights_norm * torch.sum(distributions_per_traj*torch.log(distributions_per_traj), dim=tuple(range(1, len(distributions_per_traj.shape)))), dim=0)
    # SINGLE AGENT IMPORTANCE SAMPLING
    entropy_a1_norm = - torch.sum(importance_weights_norm_a1 * torch.sum(distributions_per_traj_a1*torch.log(distributions_per_traj_a1), dim=tuple(range(1, len(distributions_per_traj_a1.shape)))), dim=0)
    entropy_a2_norm = - torch.sum(importance_weights_norm_a2 * torch.sum(distributions_per_traj_a2*torch.log(distributions_per_traj_a2), dim=tuple(range(1, len(distributions_per_traj_a2.shape)))), dim=0)

    return entropy_norm, entropy_a1_norm, entropy_a2_norm


def compute_entropy_knn(env, behavioral_policies, target_policies, states, actions,
                    num_traj, real_traj_lengths, distances, indices, k, G, B, ns, eps):
    importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2 = compute_importance_weights_knn(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    # Compute objective function
    # compute weights sum for each particle
    a12_indices, a12_distances = torch.tensor(indices[0], dtype=int_type), torch.tensor(distances[0], dtype=float_type)
    a1_indices, a1_distances = torch.tensor(indices[1], dtype=int_type), torch.tensor(distances[1], dtype=float_type)
    a2_indices, a2_distances = torch.tensor(indices[2], dtype=int_type), torch.tensor(distances[2], dtype=float_type)
    weights_sum = torch.sum(importance_weights_norm[a12_indices[:, :-1]], dim=1)
    # compute volume for each particle
    volumes = (torch.pow(a12_distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
    # compute entropy
    entropy = - torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B
    weights_sum_a1 = torch.sum(importance_weights_norm_a1[a1_indices[:, :-1]], dim=1)
    # compute volume for each particle
    volumes_a1 = (torch.pow(a1_distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
    # compute entropy
    entropy_a1 = - torch.sum((weights_sum_a1 / k) * torch.log((weights_sum_a1 / (volumes_a1 + eps)))) + B

    weights_sum_a2 = torch.sum(importance_weights_norm_a2[a2_indices[:, :-1]], dim=1)
    # compute volume for each particle
    volumes_a2 = (torch.pow(a2_distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
    # compute entropy
    entropy_a2 = - torch.sum((weights_sum_a2 / k) * torch.log((weights_sum_a2 / (volumes_a2 + eps)))) + B

    return entropy, entropy_a1, entropy_a2


def compute_mutual_information(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths):
    _, importance_weights_norm, importance_weights_norm_a1, importance_weights_norm_a2  = compute_importance_weights(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    distributions_per_traj, distributions_per_traj_a1, distributions_per_traj_a2 = compute_distributions(env, states, num_traj, real_traj_lengths)
    distributions_per_traj_a1_exp = distributions_per_traj_a1.unsqueeze(3).unsqueeze(4)
    distributions_per_traj_a2_exp = distributions_per_traj_a2.unsqueeze(1).unsqueeze(1)
    mi = torch.sum(importance_weights_norm * torch.sum(distributions_per_traj*(torch.log(distributions_per_traj) - torch.log(distributions_per_traj_a1_exp) - torch.log(distributions_per_traj_a2_exp)), dim=(1,2,3,4,5,6)), dim=0)
    return mi.detach().clone(), mi.detach().clone()


def compute_kl(env, behavioral_policies, target_policies, states):
    # numpy_states = torch.from_numpy(states).type(float_type)
    numeric_error = False
    kls = []
    # Compute KL divergence between behavioral and target policy
    for idx, (behavioral_policy, target_policy) in enumerate(zip(behavioral_policies, target_policies)):
        p0, _, _ = behavioral_policy.forward(states) if not behavioral_policy.policy_decentralized else behavioral_policy.forward(states[:,:,env.state_indeces[idx]])
        p1, _, _ = target_policy.forward(states) if not target_policy.policy_decentralized else target_policy.forward(states[:,:,env.state_indeces[idx]])
        kl = torch.sum(p0*(torch.log(p0) -torch.log(p1)), dim=(0,1)).mean()
        kls.append(kl)
        numeric_error = torch.isinf(kl) or torch.isnan(kl) or numeric_error
    # Minimum KL is zero
    kls = [torch.max(torch.tensor(0.0), kl) for kl in kls]
    return kls, numeric_error

def compute_kl_knn(env, behavioral_policies, target_policies, states, actions,
               num_traj, real_traj_lengths, distances, indices, k, eps):
    kls = []
    numeric_error = False
    _, importance_weights_norm_a1, importance_weights_norm_a2 = compute_importance_weights_knn(env, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths)
    a1_indices, _ = torch.tensor(indices[1], dtype=int_type), torch.tensor(distances[1], dtype=float_type)
    a2_indices, _ = torch.tensor(indices[2], dtype=int_type), torch.tensor(distances[2], dtype=float_type)

    weights_sum_a1 = torch.sum(importance_weights_norm_a1[a1_indices[:, :-1]], dim=1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights_norm_a1.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum_a1) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl) or numeric_error
    kls.append(torch.max(torch.tensor(0.0), kl))
    weights_sum_a2 = torch.sum(importance_weights_norm_a2[a2_indices[:, :-1]], dim=1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights_norm_a2.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum_a2) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl) or numeric_error
    kls.append(torch.max(torch.tensor(0.0), kl))

    return kls, numeric_error

def log_epoch_statistics(writer, log_file, csv_file_1, csv_file_2, epoch,
                         loss, entropy, num_off_iters, execution_time, full_entropy,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss A1", loss[0], global_step=epoch)
    writer.add_scalar("Loss A2", loss[1], global_step=epoch)
    writer.add_scalar("Entropy", entropy[0], global_step=epoch)
    writer.add_scalar("Entropy A1", entropy[1], global_step=epoch)
    writer.add_scalar("Entropy A2", entropy[2], global_step=epoch)
    writer.add_scalar("MI A12", entropy[3], global_step=epoch)
    writer.add_scalar("MI A21", entropy[4], global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)
    if full_entropy is not None:
        writer.add_scalar(f"Full Entropy:", full_entropy, global_step=epoch)

    if heatmap_image is not None:
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)
        writer.add_scalar("Exact Entropy", heatmap_entropy[0], global_step=epoch)
        writer.add_scalar("Exact Entropy A1", heatmap_entropy[1], global_step=epoch)
        writer.add_scalar("Exact Entropy A2", heatmap_entropy[2], global_step=epoch)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["Entropy", fancy_float(entropy[0])],
        ["Entropy A1", fancy_float(entropy[1])],
        ["Entropy A2", fancy_float(entropy[2])],
        ["Off-policy iters", num_off_iters]
    ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch}, {loss[0]}, {loss[1]}, {entropy[0]}, {entropy[1]}, {entropy[2]}, {entropy[3]},  {entropy[4]}, {full_entropy}, {num_off_iters}, {execution_time}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy[0]}, {heatmap_entropy[1]}, {heatmap_entropy[2]}\n")
        csv_file_2.flush()
        table.extend([
            ["Exact Entropy", fancy_float(heatmap_entropy[0])],
            ["Exact Entropy A1", fancy_float(heatmap_entropy[1])],
            ["Exact Entropy A2", fancy_float(heatmap_entropy[2])]
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
    entropy, entropy_a1, entropy_a2 = compute_entropy(env, behavioral_policies, target_policies, states, actions, num_traj, traj_len)
    mi_a12, mi_a21 = compute_mutual_information(env, behavioral_policies, target_policies, states, actions, num_traj, traj_len)
    if update_algo == "Centralized":
        losses = - entropy
        numeric_error = torch.isinf(losses) or torch.isnan(losses)
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].zero_grad()
        losses.backward()
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].step()
    elif update_algo == "Decentralized":
        losses = [- entropy_a1, - entropy_a2]
        numeric_error = torch.isinf(losses[0]) or torch.isinf(losses[1]) or torch.isnan(losses[0]) or torch.isnan(losses[1])
        conditions = [not v for v in thresholds if not v]
        last_opt_idx = max((i for i, c in enumerate(conditions) if c), default=-1)
        for idx, (opt, loss, should_opt) in enumerate(zip(optimizers, losses, conditions)):
            if should_opt:
                try:
                    opt.zero_grad()
                    loss.backward(retain_graph=(idx != last_opt_idx))
                    opt.step()
                except RuntimeError as e:
                    print(f"Error optimizing at index {idx}: {e}")
                    raise
    elif update_algo == "Decentralized_MI":
        losses = [- entropy_a1 + mi_a12 , - entropy_a2 + mi_a21]
        numeric_error = torch.isinf(losses[0]) or torch.isinf(losses[1]) or torch.isnan(losses[0]) or torch.isnan(losses[1]) 
        conditions = [not v for v in thresholds if not v]
        last_opt_idx = max((i for i, c in enumerate(conditions) if c), default=-1)
        for idx, (opt, loss, should_opt) in enumerate(zip(optimizers, losses, conditions)):
            if should_opt:
                try:
                    opt.zero_grad()
                    loss.backward(retain_graph=(idx != last_opt_idx))
                    opt.step()
                except RuntimeError as e:
                    print(f"Error optimizing at index {idx}: {e}")
                    raise

    return losses, numeric_error

#TODO ADD MUTUAL INFO
def policy_update_knn(env, thresholds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, traj_len, update_algo, distances, indices, k, G, B, ns, eps):
    # Maximize entropy
    entropy, entropy_a1, entropy_a2 = compute_entropy_knn(env, behavioral_policies, target_policies, states, actions, num_traj, traj_len, distances, indices, k, G, B, ns, eps)
    mi_a12, mi_a21 = 0, 0
    if update_algo == "Centralized":
        losses = - entropy
        numeric_error = torch.isinf(losses) or torch.isnan(losses)
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].zero_grad()
        losses.backward()
        for id in [i for i, v in enumerate(thresholds) if not v]:
            optimizers[id].step()
    elif update_algo == "Decentralized":
        losses = [- entropy_a1, - entropy_a2]
        numeric_error = torch.isinf(losses[0]) or torch.isinf(losses[1]) or torch.isnan(losses[0]) or torch.isnan(losses[1])
        conditions = [not v for v in thresholds if not v]
        last_opt_idx = max((i for i, c in enumerate(conditions) if c), default=-1)
        for idx, (opt, loss, should_opt) in enumerate(zip(optimizers, losses, conditions)):
            if should_opt:
                try:
                    opt.zero_grad()
                    loss.backward(retain_graph=(idx != last_opt_idx))
                    opt.step()
                except RuntimeError as e:
                    print(f"Error optimizing at index {idx}: {e}")
                    raise
    elif update_algo == "Decentralized_MI":
        losses = [- entropy_a1 + mi_a12, - entropy_a2 + mi_a21]
        numeric_error = torch.isinf(losses[0]) or torch.isinf(losses[1]) or torch.isnan(losses[0]) or torch.isnan(losses[1]) 
        conditions = [not v for v in thresholds if not v]
        last_opt_idx = max((i for i, c in enumerate(conditions) if c), default=-1)
        for idx, (opt, loss, should_opt) in enumerate(zip(optimizers, losses, conditions)):
            if should_opt:
                try:
                    opt.zero_grad()
                    loss.backward(retain_graph=(idx != last_opt_idx))
                    opt.step()
                except RuntimeError as e:
                    print(f"Error optimizing at index {idx}: {e}")
                    raise

    return losses, numeric_error


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
        # env.reset(seed)

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policies = []
    target_policies = []
    last_valid_target_policies = []
    for agent in range(env.n_agents):
        behavioral_policies.extend([create_policy(update_algo, is_behavioral=True)])
        target_policies.extend([create_policy(update_algo)])
        last_valid_target_policies.extend([create_policy(update_algo)])
        target_policies[agent].load_state_dict(behavioral_policies[agent].state_dict())
        last_valid_target_policies[agent].load_state_dict(behavioral_policies[agent].state_dict())

    # Set optimizer
    optimizers = []
    if optimizer == 'rmsprop':
        for agent in range(env.n_agents):
            optimizers.extend([torch.optim.RMSprop(target_policies[agent].parameters(), lr=learning_rate)])
    elif optimizer == 'adam':
        for agent in range(env.n_agents):
            optimizers.extend([torch.optim.Adam(target_policies[agent].parameters(), lr=learning_rate)])
    else:
        raise NotImplementedError
    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")
    csv_file_1 = open(os.path.join(out_path, f"{env_name}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss A1', 'loss A2', 'entropy', 'entropy A1', 'entropy A2',  'MI A12', 'MI A21', 'full_entropy', 'num_off_iters','execution_time']))
    csv_file_1.write("\n")

    if heatmap_discretizer is not None:
        csv_file_2 = open(os.path.join(out_path, f"{env_name}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'exact_entropy', 'exact_entropy_a1', 'exact_entropy_a2']))
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

    if env.discrete:
        # Discrete Entropy 
        states, actions, real_traj_lengths, next_states, _, _ = collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers, None, None)

        with torch.no_grad():
            entropy, entropy_a1, entropy_a2 = compute_entropy(env, behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths)
            mi_a12, mi_a21 = compute_mutual_information(env, behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths)
    else:
        # Continuous Entropy
        states, actions, real_traj_lengths, next_states, distances, indices = \
            collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers,  state_filter, k)

        with torch.no_grad():
            entropy, entropy_a1, entropy_a2 = compute_entropy_knn(env, behavioral_policies, behavioral_policies, states, actions, num_traj, real_traj_lengths, distances, indices, k, G, B, ns, eps)
            mi_a12, mi_a21 = 0, 0


    execution_time = time.time() - t0
    entropy = entropy.numpy()

    if update_algo == "Centralized":
        loss = [- entropy, -entropy]
    elif update_algo == "Decentralized":
        loss = [- entropy_a1.numpy(), -entropy_a2.numpy()]
    elif update_algo == "Decentralized_MI":
        loss = [- entropy_a1.numpy() + mi_a12.numpy(), -entropy_a2.numpy() + mi_a21.numpy()]
    else:
        raise NotImplementedError

    # Heatmap
    if heatmap_discretizer is not None:
        _, heatmap_entropies, heatmap_image = get_heatmap(env, behavioral_policies, heatmap_discretizer, num_traj, traj_len, heatmap_cmap, heatmap_interp, heatmap_labels)
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
            entropy=[entropy, entropy_a1, entropy_a2, mi_a12, mi_a21],
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

        if env.discrete:
            # Collect particles to optimize off policy
            states, actions, real_traj_lengths, next_states, _, _ = collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers, None, None)
        else: 
            # Collect particles to optimize off policy
            states, actions, real_traj_lengths, next_states, distances, indices = \
                    collect_particles_parallel(env, behavioral_policies, num_traj, traj_len, num_workers, state_filter, k)

        if use_backtracking:
            learning_rate = original_lr
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            backtrack_iter = 1
        else:
            backtrack_iter = None

        while not all(kl_threshold_reacheds):
            if env.discrete:
                # Optimize policy
                loss, numeric_error = policy_update(env, kl_threshold_reacheds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths, update_algo)
            else:
                # Optimize policy
                loss, numeric_error = policy_update_knn(env, kl_threshold_reacheds, optimizers, behavioral_policies, target_policies, states, actions, num_traj, real_traj_lengths, update_algo, distances, indices, k, G, B, ns, eps)

            if update_algo == "Centralized":
                entropy= [- loss.detach().numpy(), -loss.detach().numpy()]
            else:
                entropy= [- loss[0].detach().numpy(), -loss[1].detach().numpy()]
            with torch.no_grad():
                if env.discrete:
                    kls, kl_numeric_error = compute_kl(env, behavioral_policies, target_policies, states)
                else:
                    kls, kl_numeric_error = compute_kl_knn(env, behavioral_policies, target_policies, states,actions, num_traj, real_traj_lengths, distances, indices, k, eps)
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
                    if env.discrete:
                        entropy, entropy_a1, entropy_a2 = compute_entropy(env, last_valid_target_policies, last_valid_target_policies, states, actions, num_traj, real_traj_lengths)
                        mi_a12, mi_a21 = compute_mutual_information(env, last_valid_target_policies, last_valid_target_policies, states, actions, num_traj, real_traj_lengths)
                    else:
                        entropy, entropy_a1, entropy_a2  = compute_entropy_knn(env, last_valid_target_policies, last_valid_target_policies,states, actions, num_traj, real_traj_lengths, distances, indices, k, G, B, ns, eps)
                        mi_a12, mi_a21 = 0, 0

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
                    elif update_algo == "Decentralized_MI":
                        loss = [- entropy_a1.numpy() + mi_a12.numpy(), -entropy_a2.numpy() + mi_a21.numpy()]
                    else:
                        raise NotImplementedError
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None:
                            _, heatmap_entropies, heatmap_image = \
                                get_heatmap(env, behavioral_policies, heatmap_discretizer, num_traj, traj_len, heatmap_cmap, heatmap_interp, heatmap_labels)
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
                        entropy=[entropy, entropy_a1, entropy_a2, mi_a12, mi_a21],
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        full_entropy=None,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropies,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=learning_rate
                    )

    return behavioral_policies