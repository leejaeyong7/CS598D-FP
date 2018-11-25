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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.PER_e = 0.01
        self.PER_a = 0.6
        self.PER_b = 0.4
        self.PER_b_incr = 0.001
        self.abs_err_upper = 1.0

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.memory = []
        self.position = 0
        self.num_pushed = 0

    def get_leaf_node(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if (left_child_index >= len(self.tree)):
                leaf_index = parent_index
                break
            else:
                if(value <= self.tree[left_child_index]):
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        # return tree index, priority value, and actual data
        return leaf_index, self.tree[leaf_index], self.memory[data_index]

    def push(self, data):
        # finds maximum priority, and store data with max priority
        max_p = np.max(self.tree[-self.capacity:])

        # prevents 0 probability
        if(max_p == 0):
            max_p = self.abs_err_upper

        self.add_node_to_tree(max_p, data)
        self.num_pushed += 1

    def sample(self, batch_size):
        indices = np.empty((batch_size,), dtype=np.int32)
        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        memory = (states, actions, next_states, rewards, dones)
        weights = np.empty((batch_size, 1), dtype=np.float32)

        total_priority = self.get_total_priority()
        priority_segment = total_priority / batch_size

        # increment PER_b
        self.PER_b = np.min([1.0, self.PER_b + self.PER_b_incr])

        p_min = np.min(self.tree[-self.capacity:]) / total_priority
        max_weight = (p_min * batch_size) ** (-self.PER_b)

        for i in range(batch_size):
            sample_start = priority_segment * i
            sample_end = priority_segment * (i+1)

            sample_value = np.random.uniform(sample_start, sample_end)
            index, priority, data = self.get_leaf_node(sample_value)

            sample_prob = priority / total_priority

            weight = np.power(batch_size * sample_prob, -self.PER_b) / max_weight

            indices[i] = index
            state, action, next_state, reward, done = data
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            weights[i] = weight

        return indices, memory, weights

    def get_total_priority(self):
        return self.tree[0]

    def add_node_to_tree(self, priority_value, data):
        tree_id = self.position + self.capacity - 1
        self.memory[self.position] = data
        self.update_tree_node(tree_id, priority_value)

        # update position
        self.position += 1
        if(self.position >= self.capacity):
            self.position = 0

    def update_tree_node(self, tree_id, priority_value):
        # obtain change in curr value
        change = priority_value - self.tree[tree_id]
        self.tree[tree_id] = priority_value

        # propagate tree upward
        while(tree_id != 0):
            tree_id = (tree_id - 1) // 2
            self.tree[tree_id] += change

    def update_tree_nodes(self, tree_ids, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        priority_values = np.power(clipped_errors, self.PER_a)

        for tree_id, priority_value in zip(tree_ids, priority_values):
            self.update_tree_node(tree_id, priority_value)

    def __len__(self):
        return self.num_pushed
