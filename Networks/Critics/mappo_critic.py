import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, value_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, value_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)

        return value