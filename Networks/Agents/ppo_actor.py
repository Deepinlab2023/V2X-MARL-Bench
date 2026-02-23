import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

def _mask_logits_with_queue(logits: th.Tensor, queue):
    """
    Match your MAPPO/IPPO semantics:
      - If queue <= 0 => all actions except the LAST one are invalid.
      - Works for:
          * logits shape [A] and scalar queue
          * logits shape [B, A] and queue shape [B] or scalar
    """
    if queue is None:
        return logits

    invalid_mask = th.zeros_like(logits, dtype=th.bool)

    if not th.is_tensor(queue):
        if queue <= 0:
            invalid_mask[..., :-1] = True
    else:
        if queue.dim() == 0:
            if queue.item() <= 0:
                invalid_mask[..., :-1] = True
        elif queue.dim() == 1:
            done = (queue <= 0).unsqueeze(-1)  # [B, 1]
            invalid_mask[..., :-1] = done.expand_as(invalid_mask[..., :-1])
        else:
            raise ValueError(f"Unexpected queue shape: {queue.shape}")

    return logits.masked_fill(invalid_mask, -1e8)


class PPOSharedActor(nn.Module):

    def __init__(self, base_input_dim: int, action_dim: int, hidden_dim: int, num_agents: int):
        super().__init__()
        self.action_dim = action_dim
        self.num_agents = num_agents

        self.fc1 = nn.Linear(base_input_dim + num_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state_or_obs: th.Tensor, agent_id: th.Tensor) -> th.Tensor:
        # Allow single sample input
        single = False
        if state_or_obs.dim() == 1:
            state_or_obs = state_or_obs.unsqueeze(0)
            single = True

        if agent_id.dim() == 1:
            agent_id = agent_id.unsqueeze(0)

        x = th.cat([state_or_obs, agent_id], dim=-1)
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        logits = self.fc3(x)

        if single:
            logits = logits.squeeze(0)
        return logits

    def action_sampler(self, logits: th.Tensor, queue=None):

        logits = _mask_logits_with_queue(logits, queue)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist