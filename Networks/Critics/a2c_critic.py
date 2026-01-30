import torch as th
import torch.nn.functional as F


class _BaseA2CCritic(th.nn.Module):
    def __init__(self, input_dim, hidden_dim, value_dim, use_rnn: bool):
        super().__init__()
        self.rnn = use_rnn
        self.fc1 = th.nn.Linear(input_dim, hidden_dim)
        if self.rnn:
            self.gru = th.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.fc2 = th.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = th.nn.Linear(hidden_dim, value_dim)

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
            value = self.fc3(x)

        elif x.dim() == 3:
            x = F.relu(self.fc1(x))
            if self.rnn:
                x, hidden_state = self.gru(x, hidden_state)
            else:
                x = F.relu(self.fc2(x))
            value = self.fc3(x)

        else:
            raise ValueError(f"Unexpected critic input shape: {x.shape}")

        if single_sample:
            value = value.squeeze(0)

        return (value, hidden_state) if self.rnn else value


class A2CCentralizedCritic(_BaseA2CCritic):
    """
    CTDE critic for MAA2C.
    """
    def __init__(self, params):
        state_dim = params.state_dim
        hidden_dim = params.critic_hidden_dim
        value_dim = params.value_dim

        if params.prev_action_input:
            input_dim = state_dim + params.n_agent * params.action_dim
        else:
            input_dim = state_dim

        super().__init__(input_dim, hidden_dim, value_dim, params.rnn)


class A2CSharedCritic(_BaseA2CCritic):
    """
    Shared per-agent critic (IA2C parameter-sharing).

    FNN input: [base, agent_id]
    RNN input: [base, agent_id] (+ prev_action if enabled)
    """
    def __init__(self, base_state_dim, params):
        hidden_dim = params.critic_hidden_dim
        value_dim = params.value_dim

        input_dim = base_state_dim + params.n_agent
        if params.prev_action_input:
            input_dim += params.action_dim

        super().__init__(input_dim, hidden_dim, value_dim, params.rnn)


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