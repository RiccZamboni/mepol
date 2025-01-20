import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from datetime import datetime
from src.envs.rooms import Rooms
from src.algorithms.matrpo import matrpo
from src.policy import GaussianPolicy, DiscretePolicy, train_supervised


parser = argparse.ArgumentParser(description='Goal-Based Reinforcement Learning - MATRPO')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting samples')
parser.add_argument('--env', type=str, required=True,
                    help='The MDP')
parser.add_argument('--policy_init', type=str, default=None,
                    help='Path to the weights for custom policy initialization.')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of training epochs')
parser.add_argument('--batch_size', type=int, required=True,
                    help='The batch size')
parser.add_argument('--traj_len', type=int, required=True,
                    help='The maximum length of a trajectory')
parser.add_argument('--gamma', type=float, default=0.995,
                    help='The discount factor')
parser.add_argument('--lambd', type=float, default=0.98,
                    help='The GAE lambda')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='The optimizer used for the critic, either adam or lbfgs')
parser.add_argument('--critic_lr', type=float, default=1e-2,
                    help='Learning rate for critic optimization')
parser.add_argument('--critic_reg', type=float, default=1e-3,
                    help='Regularization coefficient for critic optimization')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic full updates')
parser.add_argument('--critic_batch_size', type=int, default=64,
                    help='Mini batch in case of adam optimizer for critic optimization')
parser.add_argument('--cg_iters', type=int, default=10,
                    help='Conjugate gradient iterations')
parser.add_argument('--cg_damping', type=float, default=0.1,
                    help='Conjugate gradient damping factor')
parser.add_argument('--kl_thresh', type=float, required=True,
                    help='KL threshold')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='goal_rl',
                    help='The tensorboard directory under which the directory of this experiment is put')

args = parser.parse_args()

"""
Experiments specifications

    - env_create : callable that returns the target environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy

"""
exp_spec = {
    # Multi-Agent Environments
    'Room': {
        'env_create': lambda: Rooms(H=1000, grid_size=10, n_actions=4, n_agents=2),
        'discretizer_create': lambda env: True,
        'hidden_sizes': [64, 64],
        'activation': nn.ReLU,
        'state_filter': None,
        'eps': None,
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', 'Y')
    },

}

spec = exp_spec.get(args.env)

if spec is None:
    print(f"Experiment name not found. Available ones are: {', '.join(key for key in exp_spec)}.")
    exit()

env = spec['env_create']()


def create_policy():
    policy = DiscretePolicy(
        num_features=env.num_features_per_agent,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.n_actions,
        activation=spec['activation'],
        decentralized = True
    )
    return policy

vfuncs = []
for agent in range(env.n_agents):
    # Create a critic
    hidden_sizes = [64, 64]
    hidden_activation = nn.ReLU
    layers = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            layers.extend([
                nn.Linear(env.num_features_per_agent, hidden_sizes[i]),
                hidden_activation()
            ])
        else:
            layers.extend([
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
                hidden_activation()
            ])

    layers.append(nn.Linear(hidden_sizes[i], 1))
    vfunc = nn.Sequential(*layers)

    for module in vfunc:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
    vfuncs.extend([vfunc])

policies = []
for agent in range(env.n_agents):
    if args.policy_init is not None:
        kind = 'MAMEPOLInit'
        policy = create_policy()
        policy_path = "/Users/riccardozamboni/Documents/PhD/Git/mepol/pretrained/" + args.policy_init + "-" + str(agent)
        current_directory = os.path.dirname(os.path.abspath(__file__))
        policy.load_state_dict(torch.load(policy_path))
        policies.extend([policy])
    else:
        kind = 'RandomInit'
        policy = create_policy()
        policies.extend([policy])


exp_name = f"env={args.env}_{kind}"

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results/goal_rl",
                        args.tb_dir_name, exp_name +
                        "__" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                        "__" + str(os.getpid()))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write("Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write("{}={}\n".format(key, value))

    f.write("-"*10 + "\n")

    f.write(policies[0].__str__())
    f.write("-"*10 + "\n")
    f.write(vfunc.__str__())

    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write("Setting random seed {}\n".format(args.seed))

matrpo(
    env=env,
    env_name=args.env,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    traj_len=args.traj_len,
    gamma=args.gamma,
    lambd=args.lambd,
    vfuncs=vfuncs,
    policies=policies,
    optimizer=args.optimizer,
    critic_lr=args.critic_lr,
    critic_reg=args.critic_reg,
    critic_iters=args.critic_iters,
    critic_batch_size=args.critic_batch_size,
    cg_iters=args.cg_iters,
    cg_damping=args.cg_damping,
    kl_thresh=args.kl_thresh,
    num_workers=args.num_workers,
    out_path=out_path,
    seed=args.seed
)