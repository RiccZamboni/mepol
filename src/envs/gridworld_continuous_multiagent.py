import gym
import numpy as np

# Suppress pygame welcome print
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame


class BoundingBox:
    """
    2d bounding box.
    """
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def check_if_inside(self, x_1, y_1, x_2, y_2):
        """
        Checks whether point (x,y) is inside the bounding box.
        """
        return self.xmin <= x_1 <= self.xmax and self.ymin <= y_1 <= self.ymax and self.xmin <= x_2 <= self.xmax and self.ymin <= y_2 <= self.ymax

    def rect(self, tx_1, ty_1, tx_2, ty_2, scale):
        """
        Returns a ready to render rect given translation and scale factors: (top left coordinates, width, height).
        """
        return (self.xmin*scale + tx_1, self.ymin*scale + ty_1, self.xmin*scale + tx_2, self.ymin*scale + ty_2, (self.xmax - self.xmin)*scale, (self.ymax - self.ymin)*scale)


class GridWorldContinuous(gym.Env):

    def __init__(self, dim=6, max_delta=0.2, wall_width=2.5):
        # (x1, y1, x2, y2)
        self.n_agents = 2
        self.num_features = 4
        self.num_features_per_agent = 2
        self.n_actions = 2
        self.action_dim = 4
        self.state_indeces = [[0,1], [2,3]]
        self.action_indeces = [[0,1], [2,3]]
        self.discrete = False

        # The gridworld bottom left corner is at (-self.dim, -self.dim)
        # and the top right corner is at (self.dim, self.dim)
        self.dim = dim

        # The maximum change in position obtained through an action
        self.max_delta = max_delta

        # Maximum (dx_1, dy_1, dx2, dy_2) action
        self.max_action = np.array([self.max_delta, self.max_delta, self.max_delta, self.max_delta], dtype=np.float32)
        self.action_space = gym.spaces.Box(-self.max_action, self.max_action, dtype=np.float32)

        # Maximum (x_1,y_1, x_2, y_2) position
        self.max_position = np.array([self.dim, self.dim, self.dim, self.dim], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-self.max_position, self.max_position, dtype=np.float32)

        # Current state
        self.state = None

        # Initial states of the two agents
        init_top_left = np.array([-self.dim, -self.dim, -self.dim, -self.dim], dtype=np.float32)
        init_bottom_right = np.array([-self.dim+2, -self.dim+2, -self.dim+3, -self.dim+3], dtype=np.float32)
        self.init_states = gym.spaces.Box(init_top_left, init_bottom_right, dtype=np.float32)

        self.wall_width = wall_width

        # The area in which the agent can move is defined by the gridworld box (defined by self.dim) and by the following walls
        self.walls = [
            # Central walls
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=-self.wall_width, ymax=self.wall_width),
            BoundingBox(xmin=-self.wall_width, xmax=-self.wall_width/2, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=self.wall_width/2, xmax=self.wall_width, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            # External walls
            BoundingBox(xmin=-self.dim, xmax=-(self.dim - self.wall_width), ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=-self.dim, ymax=-(self.dim - self.wall_width)),
            BoundingBox(xmin=self.dim - self.wall_width, xmax=self.dim, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=self.dim-self.wall_width, ymax=self.dim)
        ]

        # Render stuff
        self.game_display = None
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        self.SCALE = 30
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.AGENT_RADIUS = 5

    def reset(self):
        # Start near the bottom left corner of the bottom left room
        self.state = self.init_states.sample()

        # Reset pygame
        self.game_display = None

        return self.state

    def render(self):
        if self.game_display is None:
            pygame.init()
            self.game_display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
            pygame.display.set_caption('Continuous Gridworld')

        # Draw background
        self.game_display.fill(self.WHITE)

        # Draw walls
        for bbox in self.walls:
            pygame.draw.rect(self.game_display, self.BLUE, bbox.rect(int(self.DISPLAY_WIDTH/2), int(self.DISPLAY_HEIGHT/2), self.SCALE))

        xmin = -self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        xmax = self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        ymin = -self.dim*self.SCALE + self.DISPLAY_HEIGHT/2
        ymax = self.dim*self.SCALE + self.DISPLAY_HEIGHT/2

        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymin), (xmin, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymax), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmax, ymin), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymin), (xmax, ymin))

        # Draw agent
        # Take agent (x,y), change y sign, scale and translate
        agent1_x, agent1_y = self.state[0,1] * np.array([1, -1], dtype=np.int32) * self.SCALE + np.array([self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2], dtype=np.float32)
        agent2_x, agent2_y = self.state[2,3] * np.array([1, -1], dtype=np.int32) * self.SCALE + np.array([self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2], dtype=np.float32)
        pygame.draw.circle(self.game_display, self.RED, (int(agent1_x), int(agent1_y)), self.AGENT_RADIUS)
        pygame.draw.circle(self.game_display, self.RED, (int(agent2_x), int(agent2_y)), self.AGENT_RADIUS)

        # Update screen
        pygame.display.update()

    def step(self, action):
        # assert action.shape == self.action_space.shape

        x1, y1, x2, y2 = self.state

        dx1 = action[0]
        dx1 = np.clip(dx1, -self.max_delta, self.max_delta)

        dy1 = action[1]
        dy1 = np.clip(dy1, -self.max_delta, self.max_delta)

        dx2 = action[2]
        dx2 = np.clip(dx2, -self.max_delta, self.max_delta)

        dy2 = action[3]
        dy2 = np.clip(dy2, -self.max_delta, self.max_delta)

        new_x1 = x1 + dx1
        new_y1 = y1 + dy1
        new_x2 = x2 + dx2
        new_y2 = y2 + dy2


        # Check hit with a wall
        for bbox in self.walls:
            if bbox.check_if_inside(new_x1, new_y1, new_x2, new_y2):
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        if np.abs(new_x1) >= self.dim or np.abs(new_y1) >= self.dim or np.abs(new_x2) >= self.dim or np.abs(new_y2) >= self.dim:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

        self.state = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float32)
        done = False
        reward = 0

        return self.state, reward, done, {}
