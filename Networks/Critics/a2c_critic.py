import torch as th
import torch.nn.functional as F


class _BaseA2CCritic(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, value_dim):
        super().__init__()
        self.fc1 = th.nn.Linear(input_dim, hidden_dim)
        self.fc2 = th.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = th.nn.Linear(hidden_dim, value_dim)

    def forward(self, x):
        single_sample = False

        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)

        if single_sample:
            value = value.squeeze(0)

        return value


class A2CCentralizedCritic(_BaseA2CCritic):
    """
    CTDE critic for MAA2C.
    Uses global_state_dim since centralized critic observes full state.
    """
    def __init__(self, params):
        state_dim = params.global_state_dim
        hidden_dim = params.critic_hidden_dim
        value_dim = params.value_dim
        input_dim = state_dim

        super().__init__(input_dim, hidden_dim, value_dim)


class A2CSharedCritic(_BaseA2CCritic):
    """
    Shared per-agent critic (IA2C parameter-sharing).

    Input: [base, agent_id]
    """
    def __init__(self, base_state_dim, params):
        hidden_dim = params.critic_hidden_dim
        value_dim = params.value_dim
        input_dim = base_state_dim + params.n_agent

        super().__init__(input_dim, hidden_dim, value_dim)


class A2CCriticNS(th.nn.Module):
    def __init__(self, state_size, hidden_size, value_size):
        super().__init__()
        self.fc1 = th.nn.Linear(state_size, hidden_size)
        self.fc2 = th.nn.Linear(hidden_size, hidden_size)
        self.fc3 = th.nn.Linear(hidden_size, value_size)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)