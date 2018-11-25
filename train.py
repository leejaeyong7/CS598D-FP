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
NUM_EPISODES = 50000000

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

MEMORY_CAPACITY = 200000
memory = ReplayMemory(MEMORY_CAPACITY)


def experience_replay_update(experience, weights):
    states, actions, next_states, rewards, dones = experience
    states = torch.cat(states)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    weights = torch.tensor(weights).to(device)

    valid_next_states = [
        next_states[i]
        for i, done in enumerate(dones)
        if not done
    ]
    valid_next_masks = [not done for done in dones]

    next_states = torch.cat(valid_next_states)
    next_masks = torch.tensor(valid_next_masks,
                              device=device,
                              dtype=torch.uint8)

    # all state_action value pairs = Q(S_t, A_1...T)
    all_q = policy(states)

    # actual state_action value pairs = Q(S_t, A_t)
    best_q = all_q.gather(1, actions)
    # compute state values V(S_t+1) using target network
    with torch.no_grad():
        all_expected_q = torch.zeros(BATCH_SIZE, device=device)
        all_expected_q[next_masks] = target(next_states).max(1)[0]
        best_expected_q = (all_expected_q * REWARD_DECAY) + rewards

    q_diff = best_expected_q - best_q[:, 0]
    loss = (weights * q_diff.pow(2)).mean()
    errors = torch.abs(q_diff).detach()
    return loss, errors, rewards.mean()


# initialize
game.reset()

state, screen = game.get_state()
state = state.to(device)
print('Filling up memory')
# first fill in memory with experiences
for t in range(MEMORY_CAPACITY):
    # sample action from observed state
    action = torch.tensor([game.actions.sample()],
                          dtype=torch.long, device=device)
    obs, reward, done, info = game.apply_action(action)

    if(done):
        next_state = None
        game.reset()
    else:
        next_state, screen = game.get_state()
        next_state = next_state.to(device)
    reward_tensor = torch.tensor([reward], device=device)
    if((t % 5000) == 0):
        print("filled in {}".format(t))
    memory.push((state, action, next_state, reward_tensor, done))

policy.steps = 0

print('Training Start')
for episode in range(NUM_EPISODES):
    # initialize
    game.reset()

    state, screen = game.get_state()
    state = state.to(device)

    episode_loss = 0
    episode_reward = 0
    episode_errors = 0
    episode_update = 0

    episode_video = None
    episode_video_frames = [screen[:, 0]]
    for t in count():
        # sample action from observed state
        with torch.no_grad():
            action = policy.get_action(state)
        obs, reward, done, info = game.apply_action(action)

        # observe new state iff game is not over
        if(done):
            next_state = None
            episode_video = torch.stack(episode_video_frames, dim=2)
        else:
            next_state, screen = game.get_state()
            next_state = next_state.to(device)
            for k in range(screen.shape[0]):
                episode_video_frames.append(screen[:, k])
        reward_tensor = torch.tensor([reward]).to(device)
        memory.push((state, action, next_state, reward_tensor, done))

        # update state
        state = next_state

        # update model
        if(len(memory) >= MEMORY_CAPACITY):
            optimizer.zero_grad()

            # perform standard DQN update with experience replay
            indices, experience, weights = memory.sample(BATCH_SIZE)
            loss, errors, reward = experience_replay_update(experience,
                                                            weights)
            memory.update_tree_nodes(indices, errors)
            loss.backward()

            episode_loss += loss.item()
            episode_reward += reward.item()
            episode_errors += errors.mean().item()
            episode_update += 1
            # clip gradient
            # clip_grad_value_(policy.parameters(), GRAD_CLIP)
            optimizer.step()
        # check if game is done
        if(done):
            break
    if(episode_update != 0):
      episode_loss /= episode_update
      episode_reward /= episode_update
      episode_errors /= episode_update
    else:
      episode_loss = 0
      episode_reward = 0
      episode_errors = 0

    num_frames = len(episode_video_frames)

    writer.add_scalar('data/episode_reward', episode_reward, episode)
    writer.add_scalar('data/episode_length', num_frames, episode)

    # update target network from current policy network
    if((episode + 1) % TARGET_UPDATE == 0):
        target.load_state_dict(policy.state_dict())

    if((episode) % 30 == 0):
        writer.add_video('video/episode', episode_video, episode)

    # save model every 500 episode
    if((episode + 1) % 500 == 0):
        torch.save(policy, MODEL_PATH)

