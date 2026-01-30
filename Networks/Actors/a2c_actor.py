import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

def _mask_logits_with_queue(logits, queue):
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


class A2CSharedActor(th.nn.Module):
    """
    Shared actor (parameter sharing) for IA2C and MAA2C.

    IMPORTANT:
    - This module expects the trainer to build the input vector.
    - For FNN:   input = [base, agent_id]
    - For RNN:   input = [base, agent_id] (+ prev_action if enabled)
    """
    def __init__(self, input_dim, params):
        super().__init__()
        self.action_dim = params.action_dim
        self.hidden_dim = params.actor_hidden_dim
        self.rnn = params.rnn

        self.fc1 = th.nn.Linear(input_dim, self.hidden_dim)
        if self.rnn:
            self.gru = th.nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        else:
            self.fc2 = th.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = th.nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x, hidden_state=None):
        single_sample = False

        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True

        if x.dim() == 2:
            x = F.relu(self.fc1(x))
            if self.rnn:
                x = x.unsqueeze(1)
                x, hidden_state = self.gru(x, hidden_state)
                x = x.squeeze(1)
            else:
                x = F.relu(self.fc2(x))
            logits = self.fc3(x)

        elif x.dim() == 3:
            x = F.relu(self.fc1(x))
            if self.rnn:
                x, hidden_state = self.gru(x, hidden_state)
            else:
                x = F.relu(self.fc2(x))
            logits = self.fc3(x)

        else:
            raise ValueError(f"Unexpected actor input shape: {x.shape}")

        if single_sample:
            logits = logits.squeeze(0)

        return (logits, hidden_state) if self.rnn else logits

    def action_sampler(self, logits, queue=None):
        logits = _mask_logits_with_queue(logits, queue)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist


class A2CActorNS(th.nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.action_dim = action_size
        self.fc1 = th.nn.Linear(state_size, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, hidden_size)
        self.fc3 = th.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def action_sampler(self, logits, queue=None):
        logits = _mask_logits_with_queue(logits, queue)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist