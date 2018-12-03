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
from atari_wrappers import make_atari, wrap_deepmind
from game import Game
from replay_memory import ReplayMemory
from tensorboardX import SummaryWriter

import logging

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

BATCH_SIZE = 32
REWARD_DECAY = 0.99
GRAD_CLIP = 1
TARGET_UPDATE = 50000
NUM_FRAMES = 50000000
# LEARNING_RATE = 0.00025
LEARNING_RATE = 1e-4

MODEL_PATH = './dqn-tennis.model'

# transform = T.Compose([
#   T.ToPILImage(),
#   T.Resize((84, 84), interpolation=Image.CUBIC),
#   T.Grayscale(),
#   T.ToTensor()])

# game = Game('TennisDeterministic-v4', transform=transform, keep_frames=4)
game = make_atari('BreakoutNoFrameskip-v4')
game = wrap_deepmind(game, frame_stack=True, clip_rewards=False, pytorch_img=True)
device = torch.device('cuda:1')
num_actions = game.action_space.n
policy = DQN(num_actions=num_actions).to(device)
target = DQN(num_actions=num_actions).to(device)
target.load_state_dict(policy.state_dict())
target.eval()

writer = SummaryWriter()

# optimizer = optim.RMSprop(policy.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

MEMORY_CAPACITY = 1000000
memory = ReplayMemory(MEMORY_CAPACITY)


def calculate_loss(experience, weights):
    states, actions, next_states, rewards, dones = experience
    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device).view(-1, 1)
    next_states = torch.tensor(next_states).to(device)
    masks = torch.tensor(dones).to(device)
    weights = torch.tensor(weights).to(device)

    # all state_action value pairs = Q(S_t, A_1...T)
    all_q = policy(states)

    # actual state_action value pairs = Q(S_t, A_t)
    q = all_q.gather(1, actions.view(-1, 1))

    # compute state values V(S_t+1) using target network
    with torch.no_grad():
        best_expected_actions = policy(next_states).max(1)[1]
        all_expected_target_q = target(next_states)
        best_expected_target_q = all_expected_target_q.gather(1, best_expected_actions.view(-1, 1))
        best_expected_target_q[masks] = 0.0
        expected_target_q = (best_expected_target_q * REWARD_DECAY) + rewards

    q_diff = expected_target_q - q
    loss = (weights * q_diff.pow(2)).mean()
    errors = torch.abs(q_diff).detach()
    return loss, errors, rewards.mean()


# initialize
state = game.reset()

# state = game.get_state()
logging.info('Filling up memory')
# first fill in memory with experiences
for t in range(MEMORY_CAPACITY):
    # sample action from observed state
    action = game.actions.sample()
    next_state, reward, done, info = game.step(action)

    # next_state = game.get_state()
    memory.push((state, action, next_state, reward, done))
    state = next_state

    if(done):
        state = game.reset()
        # state = game.get_state()

    if((t % 50000) == 0):
        logging.info('finished {:.02f} %'.format(t / MEMORY_CAPACITY * 100))

logging.info('Training Start')

total_frame_count = 0
for episode in count():
    # initialize
    state = game.reset()
    policy.train()

    episode_reward = 0
    episode_update_reward = 0
    episode_update = 0

    # state = game.get_state()
    for t in count():
        # sample action from observed state
        action = policy.get_greedy_action(state)
        next_state, reward, done, info = game.step(action)
        episode_reward += reward

        # save next state
        # next_state = game.get_state()
        memory.push((state, action, next_state, reward, done))
        state = next_state

        optimizer.zero_grad()

        # perform standard DQN update with prioritized experience replay
        indices, experience, weights = memory.sample(BATCH_SIZE)
        loss, errors, reward = calculate_loss(experience, weights)
        loss.backward()
        memory.update_tree_nodes(indices, errors)

        # clip gradient
        clip_grad_value_(policy.parameters(), GRAD_CLIP)

        optimizer.step()

        # update episode graph variables
        episode_update_reward += reward.item()
        episode_update += 1
        total_frame_count += 1
        writer.add_scalar('data/loss', loss.item(), total_frame_count)
        writer.add_scalar('data/eps', policy._get_eps(), total_frame_count)

        # perform total frame count based updates
        # update target network from current policy network
        if((total_frame_count) % TARGET_UPDATE == 0):
            target.load_state_dict(policy.state_dict())
            torch.save(policy, MODEL_PATH)

        # check if game is done
        if(done or t > 3000):
            break

    episode_update_reward /= episode_update
    writer.add_scalar('data/episode_update_reward', episode_update_reward, episode)
    writer.add_scalar('data/episode_reward', episode_reward, episode)
    writer.add_scalar('data/episode_length', episode_update, episode)

    # create video every 100 episodes
    if((episode % 100) == 0):
        policy.eval()
        state = game.reset()
        episode_video_frames = []
        for t in count():
            action = policy.get_greedy_action(state, False)
            state, _, done, _ = game.step(action)
            obs = game.env.render(mode='rgb_array').transpose((2, 0, 1))
            episode_video_frames.append(obs)
            if(done or t > 3000):
                break
        # stacked with T, H, W, C
        stacked_frames = np.stack(episode_video_frames).transpose(3, 0, 1, 2)
        stacked_frames = np.expand_dims(stacked_frames, 0)
        # video takes B, C, T, H, W
        writer.add_video('video/episode', stacked_frames, episode)

    if(total_frame_count > NUM_FRAMES):
        torch.save(policy, MODEL_PATH)
        break
