import random
from typing import Union

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.layer_norm1(self.layer1(x)))
        x = F.relu(self.layer_norm2(self.layer2(x)))
        return self.layer3(x)


class QMIXAgent:
    def __init__(
        self,
        ag_idx: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        gamma: float,
        tau: float,
        hidden_dim: int,
        force_nt_when_empty: bool,
    ):
        self.ag_idx = ag_idx
        self.num_agents = num_agents

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau

        self.eps_threshold = 0.0
        self.n_episode = 0

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.force_nt_when_empty = force_nt_when_empty

    @staticmethod
    def _to_tensor(data: Union[np.ndarray, th.Tensor, float, int], device: th.device) -> th.Tensor:
        if isinstance(data, th.Tensor):
            return data.to(device=device, dtype=th.float32)
        if isinstance(data, np.ndarray):
            return th.tensor(data, dtype=th.float32, device=device)
        return th.tensor(data, dtype=th.float32, device=device)

    def select_action(self, state, env) -> th.Tensor:
        """
        Select action with optional NT constraint when queue is empty.

        Args:
            state: Current state
            env: Environment (to access queue values)

        Returns:
            action: Selected action as tensor
        """
        # Force NT when queue is empty (only applies to SIG/POSIG tasks)
        if self.force_nt_when_empty and env.queue[self.ag_idx][0] == 0:
            return th.tensor([[self.action_dim - 1]], dtype=th.long, device=self.device)

        state_t = self._to_tensor(state, self.device)

        with th.no_grad():
            q_values = self.q_net(state_t)

        if random.random() > self.eps_threshold:
            return q_values.max(1)[1].view(1, 1)

        return th.tensor([[random.randrange(self.action_dim)]], dtype=th.long, device=self.device)

    def soft_update_target_net(self) -> None:
        target_sd = self.target_net.state_dict()
        q_sd = self.q_net.state_dict()
        for k in q_sd:
            target_sd[k] = q_sd[k] * self.tau + target_sd[k] * (1 - self.tau)
        self.target_net.load_state_dict(target_sd)

    def get_action_values(self, state) -> th.Tensor:
        state_t = self._to_tensor(state, self.device)
        with th.no_grad():
            return self.q_net(state_t)