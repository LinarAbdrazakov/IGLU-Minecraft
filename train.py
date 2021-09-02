import gym
import numpy as np 
import torch
from torch import optim

from iglu.tasks import TaskSet

from wrappers import PovOnlyWrapper, IgluActionWrapper
from algorithms import DQN, ReplayBuffer

TASK = 'C17'
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('IGLUSilentBuilder-v0', max_steps=1000)
env.update_taskset(TaskSet(preset=[TASK]))
env = PovOnlyWrapper(env)
env = IgluActionWrapper(env)

num_actions = env.action_space.n
policy_net = DQN(num_actions)
target_net = DQN(num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
buffer = ReplayBuffer(10000)









