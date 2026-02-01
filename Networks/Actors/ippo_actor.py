import torch as th
import torch.nn as nn
from torch.distributions import Categorical

# Actor Parameter Sharing
class ActorPS(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, num_agents):
        super(ActorPS, self).__init__()
        self.action_dim = action_dim
        # self.state_type = args.V2I_V2V_scenario_state_type
        # self.n_pw_levels = args.num_pw_levels
        self.fc1 = nn.Linear(state_dim + num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
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
        logits = self.fc3(x)
        return logits

    def action_sampler(self, logits, queue=None):
        if queue is not None:
            invalid_mask = th.zeros_like(logits, dtype=th.bool)
            if queue <= 0:
                invalid_mask[:-1] = True
            logits.masked_fill_(invalid_mask, -1e8)
            
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action, log_prob, action_dist