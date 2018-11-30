import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

class Game(Dataset):
    def __init__(self, gym_game_type, transform, keep_frames=4):
        self.env = gym.make(gym_game_type).unwrapped
        self.keep_frames = keep_frames
        self.frames = deque([], maxlen=keep_frames)
        self.env.reset()
        self.transform = transform
        self.actions = self.env.action_space

    def get_state(self):
        return np.stack(self.frames, axis=1)

    def get_screen(self):
        return self.env.render(mode='rgb_array').transpose((2, 0, 1))

    def transform_screen(self, screen):
        return self.transform(torch.from_numpy(screen))

    def apply_action(self, action):
        obs, reward, done, _ = self.env.step(action)
        screen = self.get_screen()
        transformed = self.transform_screen(screen)
        self.frames.append(transformed)
        return obs, reward, done, _

    def reset(self):
        obs = self.env.reset()
        screen = self.get_screen()
        transformed = self.transform_screen(screen)
        for i in range(self.keep_frames):
            self.frames.append(transformed)
        return obs
