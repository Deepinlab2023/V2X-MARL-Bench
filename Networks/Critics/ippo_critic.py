import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Critic Parameter Sharing
class CriticPS(nn.Module):
    def __init__(self, state_dim, hidden_size, value_dim, num_agents):
        super(CriticPS, self).__init__()

        self.fc1 = nn.Linear(state_dim + num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, value_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, agent_id):
        x = th.cat([state, agent_id], dim=-1)
        x = th.tanh(self.fc1(x))
        x = th.tanh(self.fc2(x))
        value = self.fc3(x)
        return value