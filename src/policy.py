import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as Functional
import torch.nn.init as init

from src.utils.dtypes import float_type, int_type, eps

from collections import OrderedDict


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy with state-independent diagonal covariance matrix
    """

    def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU, decentralized= False):
        super().__init__()
        self.num_features = num_features
        self.policy_decentralized = decentralized

        self.activation = activation

        layers = []
        layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=float_type))

        # Constants
        self.log_of_two_pi = torch.tensor(np.log(2*np.pi), dtype=float_type)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mean.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def get_log_p(self, states, actions):
        mean, _ = self(states)
        return torch.sum(
            -0.5 * (
                self.log_of_two_pi
                + 2*self.log_std
                + ((actions - mean)**2 / (torch.exp(self.log_std) + eps)**2)
            ), dim=1
        )

    def forward(self, x, deterministic=False):
        mean = self.mean(self.net(x))

        if deterministic:
            output = mean
        else:
            output = mean + torch.randn(mean.size(), dtype=float_type) * torch.exp(self.log_std)

        return mean, output


    def predict(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=float_type).unsqueeze(0)
            return self(s, deterministic=deterministic)[1][0]
        

class DiscretePolicy(nn.Module):
    def __init__(self, hidden_sizes, num_features, action_dim, activation=nn.ReLU, decentralized = False):
        """
        Args:
            input_dim (int): Dimension of input state
            hidden_dims (list): List of hidden layer dimensions
            action_dims (list): List of discrete sizes for each action dimension
        """
        super(DiscretePolicy, self).__init__()

        self.activation = activation
        self.num_features = num_features
        self.policy_decentralized = decentralized
        layers = []
        linear = nn.Linear(num_features, hidden_sizes[0])
        # Initialize weights with Xavier uniform
        init.xavier_uniform_(linear.weight)
        # Initialize biases to zero
        init.zeros_(linear.bias)
        layers.extend((linear, self.activation()))
        for i in range(len(hidden_sizes) - 1):
            linear = nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
            # Initialize weights with Xavier uniform
            init.xavier_uniform_(linear.weight)
            # Initialize biases to zero
            init.zeros_(linear.bias)
            layers.extend((linear, self.activation()))

        self.network = nn.Sequential(*layers)
        
        # Single action head that outputs logits for all dimensions concatenated
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        # Initialize final layer to produce near-uniform probabilities
        init.uniform_(self.action_head.weight, -0.001, 0.001)
        init.zeros_(self.action_head.bias)
        self.action_dim = action_dim

        
    def forward(self, x):
        """
        Forward pass to compute action probabilities
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            action_probs (torch.Tensor): Action probabilities after softmax
            action (torch.Tensor): Sampled action based on probabilities
            log_prob (torch.Tensor): Log probability of sampled action
        """
        features = self.network(x)
        action_logits = self.action_head(features)
        action_probs = Functional.softmax(action_logits, dim=-1)
        
        # Sample action from categorical distribution
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        
        return action_probs, action, log_prob
    
    def predict(self, state):
        """
        Helper method to get just the action for a given state
        
        Args:
            state (torch.Tensor): Input state tensor
            
        Returns:
            action (int): Selected action index
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=float_type).unsqueeze(0)
            action_probs, action, _ = self.forward(state)
            return action
            
    def get_log_p(self, states, actions):
        """
        Get log probabilities for specific state-action pairs
        
        Args:
            states (torch.Tensor): Batch of input states
            actions (torch.Tensor): Batch of actions to evaluate
            
        Returns:
            log_probs (torch.Tensor): Log probabilities of the state-action pairs
        """
        features = self.network(states)
        action_logits = self.action_head(features)
        action_probs = Functional.softmax(action_logits, dim=-1)
        actions = torch.squeeze(actions)
        distribution = torch.distributions.Categorical(action_probs)
        log_probs = distribution.log_prob(actions)
        
        return log_probs


def train_supervised(env, policy, train_steps=100, batch_size=5000, agent_id=0):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
    dict_like_obs = True if type(env.observation_space.sample()) is OrderedDict else False

    for _ in range(train_steps):
        optimizer.zero_grad()

        if dict_like_obs:
            states = torch.tensor([env.observation_space.sample()['observation'] for _ in range(5000)], dtype=float_type)
        else:
            states = torch.tensor([env.observation_space.sample()[env.state_indeces[agent_id]] for _ in range(5000)], dtype=float_type)

        actions = policy(states)[0]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type)) ** 2)

        loss.backward()
        optimizer.step()

    return policy