import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer_norm1 = nn.LayerNorm(128)

        self.layer2 = nn.Linear(128, 128)
        self.layer_norm2 = nn.LayerNorm(128)

        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.layer1(x)))
        x = F.relu(self.layer_norm2(self.layer2(x)))
        return self.layer3(x)


class QMIXAgent:
    def __init__(self, ag_idx, num_agents, state_dim, action_dim, memory_capacity=10000, batch_size=64, gamma=0.9,
                 tau=0.005, force_nt_when_empty=True):
        self.ag_idx = ag_idx
        self.num_agents = num_agents

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.eps_threshold = 0
        self.n_episode = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.force_nt_when_empty = force_nt_when_empty

    def convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return th.tensor(data, dtype=th.float32)
        else:
            return data

    def select_action(self, state, env):
        """
        Select action with optional hard constraint enforcement when queue is empty.

        Args:
            state: Current state
            env: Environment (to access queue values)

        Returns:
            action: Selected action as tensor
        """
        # forced_joint = [9, 0, 6, 3]
        # action = forced_joint[self.ag_idx]
        # return th.tensor([[action]], dtype=th.long)
        
        # Force NT when queue is empty (only applies to SIG/POSIG tasks)
        if self.force_nt_when_empty and env.queue[self.ag_idx][0] == 0:
            action = th.tensor([[self.action_dim - 1]], dtype=th.long)
            return action

        state = self.convert_to_tensor(state).to(self.device)

        with th.no_grad():
            action_values = self.q_net(state)

        sample = random.random()

        if sample > self.eps_threshold:
            with th.no_grad():
                action = action_values.max(1)[1].view(1, 1)
        else:
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

        return action_values