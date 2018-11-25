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
class LazyFrames(object):
    '''
    reference: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L194
    '''
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class Game(Dataset):
    def __init__(self, gym_game_type, transform, keep_frames=4):
        self.env = gym.make(gym_game_type).unwrapped
        self.keep_frames = keep_frames
        self.frames = deque([], maxlen=keep_frames)
        self.env.reset()
        self.transform = transform
        self.actions = self.env.action_space

    def get_state(self):
        '''
        returns B x C x H x W array (1 x 3 x 160 x 210)
        '''
        # frames = LazyFrames(self.frames)
        transformed = torch.zeros((1, 4, 84, 84)).byte()
        screens = torch.zeros((1, 4, 3, 210, 160)).byte()
        for i in range(self.keep_frames):
          # Convert to float, rescare, convert to torch tensor
          # (this doesn't require a copy)
          # screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
          # screen = np.ascontiguousarray(self.frames[i], dtype=np.uint8)
          screen = torch.from_numpy(self.frames[i]).float()
          # Resize, and add a batch dimension (BCHW)
          if(self.transform):
              transformed[0, i]  = (self.transform(screen) * 255).byte()
          screens[0, i] = screen.byte()
        return transformed, screens

    def get_screen(self):
        return self.env.render(mode='rgb_array').transpose((2, 0, 1))

    def apply_action(self, action):
        obs, reward, done, _ = self.env.step(action.item())
        screen = self.get_screen()
        self.frames.append(screen)
        states = self.get_state()
        return states, reward, done, _

    def reset(self):
        obs = self.env.reset()
        screen = self.get_screen()
        for i in range(self.keep_frames):
            self.frames.append(screen)
