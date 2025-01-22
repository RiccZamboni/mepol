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
        self.state_indeces = [[0,1], [2,3]]
        self.action_indeces = [[0], [1]]
        self.distribution_indices = [[0,1,2,3], [0,1], [2,3]]
        self.discrete = False
        self.discretizer = None

    def set_discretizer(self, discretizer=None):
        self.discretizer = discretizer
    
    def seed(self, seed=None):
        return super().seed(seed)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        s = self.discretizer.discretize([obs[0],obs[2],obs[1],obs[3]])
        return s, reward, done, info

    def reset(self):
        obs = super().reset()
        print(obs)
        s = self.discretizer.discretize([obs[0],obs[2],obs[1],obs[3]])
        return self.discretizer.discretize([obs[0],obs[2],obs[1],obs[3]])

    def render(self, mode='human'):
        return super().render(mode)
