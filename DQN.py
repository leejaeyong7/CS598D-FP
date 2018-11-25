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
        self.steps = 0
        # 1x84x84 => 16x20x20
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        # 16x20x20 => 32x9x9
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.action_fc = nn.Linear(32*9*9, 256)
        self.state_fc = nn.Linear(32*9*9, 256)
        self.action_values = nn.Linear(256, num_actions)
        self.state_values = nn.Linear(256, 1)
        # 1x84x84 => 32x20x20
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # 32x20x20 => 64x9x9
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # 64x9x9 => 128x7x7
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(128)
        # 128*7*7 => num actions
        # self.action_fc = nn.Linear(128*7*7, 512)
        # self.state_fc = nn.Linear(128*7*7, 512)
        # self.action_values = nn.Linear(512, num_actions)
        # self.state_values = nn.Linear(512, 1)


    def to(self, device):
        super(DQN, self).to(device)
        self.device = device
        return self

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        action_value = self.action_values(self.action_fc(x))
        state_value = self.state_values(self.state_fc(x))
        q_values = state_value + action_value + action_value.mean(dim=1).view(-1, 1)
        return q_values

    def _get_eps(self):
        if(self.steps > EPS_STEP_END):
            return 0
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)

    def get_action(self, state):
        '''
        gets action based on e-greedy algorithm
        '''
        self.steps += 1
        eps = self._get_eps()
        if(random.random() > eps):
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        else:
            random_actions = random.randrange(self.num_actions)
            return torch.tensor([[random_actions]], device=self.device).long()
