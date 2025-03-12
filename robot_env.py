import gym
import numpy as np
import pygame
from gym import spaces

class RobotEnv(gym.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        pygame.init()
        self.window_size = 500
        self.robot_pos = np.array([250, 250])
        self.robot_speed = 10
        self.obstacles = [(150, 150), (350, 350), (200, 300)]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=self.window_size, shape=(2,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.robot_pos[1] -= self.robot_speed
        elif action == 1:
            self.robot_pos[0] -= self.robot_speed
        elif action == 2:
            self.robot_pos[0] += self.robot_speed

        reward = -1  
        done = False  

        for obs in self.obstacles:
            if np.linalg.norm(self.robot_pos - np.array(obs)) < 20:
                reward = -10
                done = True

        return self.robot_pos, reward, done, {}

    def reset(self):
        self.robot_pos = np.array([250, 250])
        return self.robot_pos
