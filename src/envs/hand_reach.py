import gym
import numpy as np


class HandReach(gym.Wrapper):

    def __init__(self):
        
        env = gym.make('Reacher-v4')
        super().__init__(env)
        # self.num_features = self.observation_space.sample()['observation'].shape[0]
        self.n_agents = 2
        self.num_features = 4 #only angles
        self.num_features_per_agent = 2
        self.n_actions = 1
        self.action_dim = 1
        self.state_indeces = [[0,2], [1,3]]
        self.action_indeces = [[0], [1]]
        self.distribution_indices = [[0,1,2,3], [0,2], [1,3]]
        self.discrete = False
        self.discretizer = None

    def set_discretizer(self, discretizer=None):
        self.discretizer = discretizer
    
    def seed(self, seed=None):
        return super().seed(seed)

    def step(self, action):
        obs_dict, reward, done, info = super().step(action)
        s = self.discretizer.discretize(obs_dict['joint_positions'])
        return s, reward, done, info

    def reset(self):
        obs_dict = super().reset()
        s = self.discretizer.discretize(obs_dict['joint_positions'])
        return self.discretizer.discretize(obs_dict['joint_positions'])

    def render(self, mode='human'):
        return super().render(mode)
