# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from collections import deque, namedtuple
import random
import torch
from torch import nn

from models import VisualEncoder #, TargetEncoder 

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.visual_encoder = VisualEncoder()
        self.head = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.visual_encoder(x)
        x = self.head(x)
        return x

  