import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200
EPS_STEP_END = 1000000


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        '''
        DQN network
        Assumes input of size 3 x 160 x 210
        '''
        self.num_actions = num_actions
        self.device = torch.device('cpu')


        # self.steps = 0
        # # 1x84x84 => 32x20x20
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # # 32x20x20 => 64x9x9
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # # 128*7*7 => num actions
        # self.feature_extraction = nn.Sequential(
        #     self.conv1,
        #     nn.ReLU(),
        #     self.conv2,
        #     nn.ReLU())
        # self.action_fc = nn.Linear(64* 9 * 9, 256)
        # self.state_fc = nn.Linear(64 * 9 * 9, 256)
        # self.action_values = nn.Linear(256, num_actions)
        # self.state_values = nn.Linear(256, 1)

        self.steps = 0
        # 1x84x84 => 32x20x20
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # 32x20x20 => 64x9x9
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # 64x9x9 => 128x7x7
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # 128*7*7 => num actions
        self.feature_extraction = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU())
        self.action_fc = nn.Linear(128 * 7 * 7, 512)
        self.state_fc = nn.Linear(128 * 7 * 7, 512)
        self.action_values = nn.Linear(512, num_actions)
        self.state_values = nn.Linear(512, 1)

    def to(self, device):
        super(DQN, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        x = self.feature_extraction(x).view(x.size(0), -1)
        action_v = self.action_values(self.action_fc(x))
        state_v = self.state_values(self.state_fc(x))
        return state_v + action_v - action_v.mean(dim=1).view(-1, 1)

    def _get_eps(self):
        if(self.steps > EPS_STEP_END):
            return 0
        decay_factor = math.exp(-1. * self.steps / EPS_DECAY)
        return EPS_END + (EPS_START - EPS_END) * decay_factor

    def get_greedy_action(self, state, update_step=True):
        '''
        gets action based on e-greedy algorithm
        '''
        if(update_step):
            self.steps += 1
        eps = self._get_eps()
        if(random.random() > eps):
            return self.get_action(state)
        else:
            random_actions = random.randrange(self.num_actions)
            return random_actions

    def get_action(self, state):
        dstate = torch.tensor(state).to(self.device)
        with torch.no_grad():
            return self.forward(dstate).max(1)[1].item()

