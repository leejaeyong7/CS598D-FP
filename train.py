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
from DQN import DQN
from torch.nn.utils import clip_grad_value_
from game import Game
from replay_memory import ReplayMemory
from tensorboardX import SummaryWriter

BATCH_SIZE = 32
REWARD_DECAY = 0.99
GRAD_CLIP = 1
TARGET_UPDATE = 10
NUM_FRAMES = 50000000

MODEL_PATH = './dqn.model'

transform = T.Compose([
  T.ToPILImage(),
  T.Resize((84, 84), interpolation=Image.CUBIC),
  T.Grayscale(),
  T.ToTensor()])

game = Game('BreakoutDeterministic-v4', transform=transform, keep_frames=4)
device = torch.device('cuda:1')
policy = DQN(num_actions=game.actions.n).to(device)
target = DQN(num_actions=game.actions.n).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

writer = SummaryWriter()

optimizer = optim.RMSprop(policy.parameters())

MEMORY_CAPACITY = 1000000
memory = ReplayMemory(MEMORY_CAPACITY)


def calculate_loss(experience, weights):
    states, actions, next_states, rewards, dones = experience
    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    rewards = torch.cat(rewards).to(device)
    next_states = torch.cat(next_states).to(device)
    masks = torch.cat(dones).to(device)
    weights = torch.tensor(weights).to(device)

    # all state_action value pairs = Q(S_t, A_1...T)
    all_q = policy(states)

    # actual state_action value pairs = Q(S_t, A_t)
    best_q = all_q.gather(1, actions)
    # compute state values V(S_t+1) using target network
    with torch.no_grad():
        all_expected_q = target(next_states).max(1)[0]
        all_expected_q[masks] = 0.0
        best_expected_q = (all_expected_q * REWARD_DECAY) + rewards

    q_diff = best_expected_q - best_q[:, 0]
    loss = (weights * q_diff.pow(2)).mean()
    errors = torch.abs(q_diff).detach()
    return loss, errors, rewards.mean()


# initialize
game.reset()

state, screen = game.get_state()
print('Filling up memory')
# first fill in memory with experiences
for t in range(MEMORY_CAPACITY):
    # sample action from observed state
    action = game.actions.sample()
    obs, reward, done, info = game.apply_action(action)

    if(done):
        next_state = None
    else:
        next_state, screen = game.get_state()

    memory.push((state, action, next_state, reward, done))
    state = next_state

    if(done):
        game.reset()
        state, screen = game.get_state()

    if((t % 1000) == 0):
        print('finished {}'.format(t))

print('Training Start')

total_frame_count = 0
for episode in count():
    # initialize
    game.reset()

    episode_reward = 0
    episode_update = 0

    state, screen = game.get_state()
    for t in count():
        # sample action from observed state
        action = policy.get_greedy_action(state)
        obs, reward, done, info = game.apply_action(action)

        # save next state
        next_state, screen = game.get_state()
        memory.push((state, action, next_state, reward, done))
        state = next_state

        optimizer.zero_grad()

        # perform standard DQN update with prioritized experience replay
        indices, experience, weights = memory.sample(BATCH_SIZE)
        loss, errors, reward = policy.calculate_loss(experience, weights)
        memory.update_tree_nodes(indices, errors)
        loss.backward()

        # clip gradient
        # clip_grad_value_(policy.parameters(), GRAD_CLIP)

        optimizer.step()

        # update episode graph variables
        episode_reward += reward.item()
        episode_update += 1
        total_frame_count += 1

        # perform total frame count based updates
        # update target network from current policy network
        if((total_frame_count) % TARGET_UPDATE == 0):
            target.load_state_dict(policy.state_dict())
            torch.save(policy, MODEL_PATH)

        # check if game is done
        if(done):
            break

    episode_reward /= episode_update
    writer.add_scalar('data/episode_reward', episode_reward, episode)
    writer.add_scalar('data/episode_length', episode_update, episode)

    # create video every 30 episodes
    if((episode % 30) == 0):
        obs = game.reset()
        episode_video_frames = [obs]
        for t in count(t):
            action = policy.get_action(state)
            obs, _, _, _ = game.apply_action(action)
            episode_video_frames.append(obs)
        episode_video = torch.stack(episode_video_frames)
        print(episode_video.shape)
        writer.add_video('video/episode', episode_video, episode)
