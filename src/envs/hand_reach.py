import gym
import numpy as np
from itertools import product


class HandReach(gym.Wrapper):

    def __init__(self):
        
        env = gym.make('Reacher-v4')
        super().__init__(env)
        # self.num_features = self.observation_space.sample()['observation'].shape[0]
        self.n_agents = 2
        self.num_features = 4 # only angles
        self.num_features_per_agent = 2
        self.n_actions = 1
        self.action_dim = 2
        self.state_indeces = [[0,1], [2,3]]
        self.action_indeces = [[0], [1]]
        self.distribution_indices = [[0,1,2,3], [0,1], [2,3]]
        self.a12_ind = self.distribution_indices[0]
        self.a1_ind = self.distribution_indices[1]
        self.a2_ind = self.distribution_indices[2]
        self.discrete = False
        self.discretizer = None
        self.epsilon = 0.05

    def set_discretizer(self, discretizer=None):
        self.discretizer = discretizer
        self.dim_states = tuple([self.discretizer.bins_sizes[0], self.discretizer.bins_sizes[1], self.discretizer.bins_sizes[2], self.discretizer.bins_sizes[3]]) 
        self.dim_states_a1 = tuple([self.discretizer.bins_sizes[0], self.discretizer.bins_sizes[1]])
        self.dim_states_a2 = tuple([self.discretizer.bins_sizes[2], self.discretizer.bins_sizes[3]]) 
    
    def set_seed(self, seed=None):
        self.seed = seed

    def step(self, action):
        obs_data, reward, terminated, truncated, info = super().step(action)
        # print(obs_data)
        # obs = obs_data[0]
        tip_x = obs_data[8]
        tip_y = obs_data[9]
        target_x = -0.21 # obs_data[4]
        target_y = 0.21 #obs_data[5]
        distance = np.sqrt((tip_x- target_x)**2 + (tip_y - target_y)**2)
        reward = 1. if distance <= self.epsilon else 0.
        s = self.discretizer.discretize([obs_data[0],obs_data[2],obs_data[1],obs_data[3]])
        return s, reward, terminated, info

    def reset(self):
        obs_data = self.env.reset(seed = self.seed)
        return self.discretizer.discretize([obs_data[0][0],obs_data[0][2],obs_data[0][1],obs_data[0][3]])

    def render(self, mode='human'):
        return super().render(mode)
    

    def compute_tip_position(self, sin1, cos1, sin2, cos2, l1=1.0, l2=1.0):
        # Compute sin and cos of (theta1 + theta2)
        sin12 = sin1 * cos2 + cos1 * sin2
        cos12 = cos1 * cos2 - sin1 * sin2
        
        # Compute x and y positions
        x = l1 * cos1 + l2 * cos12
        y = l1 * sin1 + l2 * sin12
        
        return x, y
    
    def compute_heatmap(self, distribution_values):
        # Initialise a 2D grid for heatmap
        bin_size = self.discretizer.bins_sizes[0]
        grid_size = bin_size ** 2
        # Link lengths
        l1, l2 = 1.0, 1.0
        # Corresponding values of cos and sin for each index (adjust as needed)
        cos_values = np.linspace(-1, 1, bin_size)  # Example range for cos
        sin_values = np.linspace(-1, 1, bin_size)  # Example range for sin
        x_range = np.linspace(-2 * l1, 2 * l2, grid_size)
        y_range = np.linspace(-2 * l1, 2 * l2, grid_size)
        heatmap = np.zeros((grid_size, grid_size))

        # Map position to grid
        def map_to_grid(x, y, x_range, y_range, grid_size):
            x_idx = np.searchsorted(x_range, x, side="right") - 1
            y_idx = np.searchsorted(y_range, y, side="right") - 1
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                return x_idx, y_idx
            return None

        # Compute the heatmap
        for i_cos1, i_sin1, i_cos2, i_sin2 in product(range(bin_size), repeat=4):
            # Joint probability for this combination
            prob = distribution_values[i_cos1, i_sin1, i_cos2, i_sin2]
            
            # Get the values of cos and sin
            cos1, sin1 = cos_values[i_cos1], sin_values[i_sin1]
            cos2, sin2 = cos_values[i_cos2], sin_values[i_sin2]
            
            # Compute sin and cos of (theta1 + theta2)
            sin12 = sin1 * cos2 + cos1 * sin2
            cos12 = cos1 * cos2 - sin1 * sin2
            
            # Compute the (x, y) position
            x = l1 * cos1 + l2 * cos12
            y = l1 * sin1 + l2 * sin12
            
            # Map the position to the heatmap grid
            grid_pos = map_to_grid(x, y, x_range, y_range, grid_size)
            if grid_pos:
                heatmap[grid_pos[1], grid_pos[0]] += prob
        heatmap /= heatmap.sum()     
        return heatmap

