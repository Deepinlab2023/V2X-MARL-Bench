import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import secrets
from Helpers.FullyObservable.idql_helper import *
from collections import namedtuple, deque
import random

device = th.device("cuda" if th.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'done', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)





class QNetwork(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer_norm1 = nn.LayerNorm(128)

        self.layer2 = nn.Linear(128, 128)
        self.layer_norm2 = nn.LayerNorm(128)

        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.layer1(x)))  # Apply layer norm then activation
        x = F.relu(self.layer_norm2(self.layer2(x)))  # Apply layer norm then activation
        return self.layer3(x)


def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0.1)  # Set weights to a fixed value
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)  # Set biases to zero

    elif isinstance(m, nn.LSTM):  # Special handling for LSTM layers
        for name, param in m.named_parameters():
            if "weight" in name:
                torch.nn.init.constant_(param, 0.1)  # Initialize all LSTM weights
            elif "bias" in name:
                torch.nn.init.constant_(param, 0.0)  # Initialize all LSTM biases



class QMIXAgent:
    def __init__(self, ag_idx, num_agents, state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.9,\
                 tau = 0.005, force_nt_when_empty=False):
        self.ag_idx = ag_idx
        self.num_agents = num_agents
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # int


        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.eps_threshold = 0
        self.n_episode = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # set_seed(123)
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        # self.q_net.apply(init_weights)
        # self.target_net.apply(init_weights)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.force_nt_when_empty = force_nt_when_empty
        

        


    def convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return th.tensor(data, dtype=th.float32)
        else:
            return data


    def select_action(self, state, game_mode, env):


        if self.force_nt_when_empty and game_mode == 2 and env.queue[self.ag_idx][0] == 0:
            action = th.tensor([[self.action_dim - 1]], dtype=th.long)
            return action


        # if self.ag_idx == 0:
        #     return th.tensor([[9]], dtype=th.long)
        # elif self.ag_idx == 1:
        #     return th.tensor([[6]], dtype=th.long)
        # elif self.ag_idx == 2:
        #     return th.tensor([[0]], dtype=th.long)
        # elif self.ag_idx == 3:
        #     return th.tensor([[3]], dtype=th.long)

        # print(self.ag_idx)

        state = self.convert_to_tensor(state).to(self.device)
        
        with th.no_grad():
            action_values = self.q_net(state)
        

        sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)


        if sample > self.eps_threshold:
            with th.no_grad():
                action = action_values.max(1)[1].view(1, 1)
        else:
            # print("self.eps_threshold: ", self.eps_threshold)
            # print("ssample           : ", sample)
            # action = th.tensor([[secrets.randbelow(self.action_dim)]], dtype=th.long)
            action = th.tensor([[random.randrange(self.action_dim)]], dtype=th.long)

        return action


    def store_transition(self, state, action, next_state, done, reward):

        state = self.convert_to_tensor(state).to(self.device)
        next_state = self.convert_to_tensor(next_state).to(self.device)
        reward = self.convert_to_tensor(reward).to(self.device)
        action = self.convert_to_tensor(action).to(self.device)

        self.memory.push(state, action, next_state, done, reward)


    def soft_update_target_net(self):

        target_net_state_dict = self.target_net.state_dict()
        q_net_state_dict = self.q_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def get_action_values(self, state):

        state = self.convert_to_tensor(state).to(self.device)

        with th.no_grad():
            action_values = self.q_net(state)

        # print("action_values: ", action_values)

        return action_values

