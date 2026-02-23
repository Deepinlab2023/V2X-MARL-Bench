# Networks/Critics/ppo_critic.py

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class _PPOBaseCritic(nn.Module):
    """
    MLP critic backbone with:
      - orthogonal init (matches your PPO critics)
      - tanh activations (matches your PPO critics)
      - single-sample friendly forward
    """
    def __init__(self, input_dim: int, hidden_dim: int, value_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, value_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _forward_mlp(self, x: th.Tensor) -> th.Tensor:
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        return self.fc3(x)

    def forward(self, x: th.Tensor) -> th.Tensor:
        single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single = True

        v = self._forward_mlp(x)

        if single:
            v = v.squeeze(0)
        return v


class PPOCentralizedCritic(_PPOBaseCritic):
    """
    Centralized critic for MAPPO:
      - FO: input = state_dim
      - POSIG non-FP: input = global_state_dim
      - POSIG FP: input = fp_critic_dim (you currently use global_state_dim + n_agent)
    """
    def __init__(self, critic_input_dim: int, params):
        super().__init__(
            input_dim=critic_input_dim,
            hidden_dim=params.critic_hidden_dim,
            value_dim=params.value_dim,
        )


class PPOSharedCritic(_PPOBaseCritic):
    """
    Shared per-agent critic for IPPO (parameter sharing critic):
      input = [base_state_or_obs, agent_id]
    This is a drop-in replacement for your old CriticPS.
    """
    def __init__(self, base_input_dim: int, params):
        super().__init__(
            input_dim=base_input_dim + params.n_agent,
            hidden_dim=params.critic_hidden_dim,
            value_dim=params.value_dim,
        )

    def forward(self, state_or_obs: th.Tensor, agent_id: th.Tensor) -> th.Tensor:
        single = False

        if state_or_obs.dim() == 1:
            state_or_obs = state_or_obs.unsqueeze(0)
            single = True

        if agent_id.dim() == 1:
            agent_id = agent_id.unsqueeze(0)

        x = th.cat([state_or_obs, agent_id], dim=-1)
        v = self._forward_mlp(x)

        if single:
            v = v.squeeze(0)
        return v