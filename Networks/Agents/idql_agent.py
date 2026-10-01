import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import secrets
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'done', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer_norm1(self.layer1(x)))
        x = F.relu(self.layer_norm2(self.layer2(x)))
        return self.layer3(x)


class GNNQNetwork(nn.Module):
    """
    Q-network of agent `agent_idx` on the SIG global state, read as a graph.

    Nodes are the agents' V2V links; node k = [g_i, g_iB, g_Bi, I_prev, queue] of agent k.
    The edge to neighbour j = [interference j -> agent_idx, interference agent_idx -> j].
    Common features = [time encoding, g_m]. One round of message passing:
        message_j = tanh(W_m [node_j, edge_j]),   h = tanh(W_u [node_self, mean_j message_j]),
    then Q = MLP([node_self, h, common]) with the same layers as QNetwork.
    `layout` gives the offsets of the state blocks (Environ.sig_state_layout()).
    """

    def __init__(self, layout, agent_idx, n_actions, hidden_dim=128, message_dim=64, gnn_hidden_dim=64):
        super(GNNQNetwork, self).__init__()
        self.layout = layout
        self.agent_idx = agent_idx
        n_agent, n_sc, sc_mult = layout["n_agent"], layout["n_sc"], layout["sc_mult"]

        # Neighbours j != agent_idx, and where each block of g_ji holds the two directions:
        # g_ji is (receiver i, interferer j != i), interferers listed in increasing order.
        self.neighbours = [j for j in range(n_agent) if j != agent_idx]
        self.out_pos = [agent_idx if agent_idx < j else agent_idx - 1 for j in self.neighbours]

        node_dim = 2 * sc_mult + 2 * n_sc + 1
        edge_dim = 2 * sc_mult
        common_dim = (layout["t"][1] - layout["t"][0]) + n_sc

        self.message = nn.Linear(node_dim + edge_dim, message_dim)
        self.update = nn.Linear(node_dim + message_dim, gnn_hidden_dim)

        self.layer1 = nn.Linear(node_dim + gnn_hidden_dim + common_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def graph_inputs(self, x):
        """Split a batch of SIG states (B, state_dim) into (nodes, edges, common)."""
        lay = self.layout
        n_agent, n_sc, sc_mult = lay["n_agent"], lay["n_sc"], lay["sc_mult"]
        batch = x.shape[0]

        def block(name, *shape):
            start, end = lay[name]
            return x[:, start:end].reshape(batch, *shape)

        nodes = th.cat((
            block("g_i", n_agent, sc_mult),
            block("g_ib", n_agent, sc_mult),
            block("g_bi", n_agent, n_sc),
            block("i_prev", n_agent, n_sc),
            block("queue", n_agent, 1),
        ), dim=-1)

        g_ji = block("g_ji", n_agent, n_agent - 1, sc_mult)
        incoming = g_ji[:, self.agent_idx]                          # j -> agent_idx
        outgoing = g_ji[:, self.neighbours, self.out_pos]           # agent_idx -> j
        edges = th.cat((incoming, outgoing), dim=-1)

        common = th.cat((block("t", -1), block("g_m", n_sc)), dim=-1)
        return nodes, edges, common

    def forward(self, x):
        nodes, edges, common = self.graph_inputs(x)
        node_self = nodes[:, self.agent_idx]

        messages = th.tanh(self.message(th.cat((nodes[:, self.neighbours], edges), dim=-1)))
        h = th.tanh(self.update(th.cat((node_self, messages.mean(dim=1)), dim=-1)))

        x = th.cat((node_self, h, common), dim=-1)
        x = F.relu(self.layer_norm1(self.layer1(x)))
        x = F.relu(self.layer_norm2(self.layer2(x)))
        return self.layer3(x)


class DQNAgent:
    def __init__(
        self,
        ag_idx: int,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        is_hysteretic_q: bool,
        memory_capacity: int,
        batch_size: int,
        gamma: float,
        tau: float,
        lr: float,
        hidden_dim: int,
        hysteretic_high_lr: float,
        hysteretic_low_lr: float,
        force_nt_when_empty: bool,
        network: str = "fc",
        state_layout=None,
        gnn_message_dim: int = 64,
        gnn_hidden_dim: int = 64,
    ):

        self.ag_idx = ag_idx
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_hysteretic_q = is_hysteretic_q

        # NT constraint
        self.force_nt_when_empty = force_nt_when_empty
        self.queue_index = state_dim - num_agents + ag_idx

        # Replay buffer
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.hysteretic_high_lr = hysteretic_high_lr
        self.hysteretic_low_lr = hysteretic_low_lr

        # Exploration
        self.eps_threshold = 0
        self.n_episode = 0

        # Device
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Networks
        if network == "gnn":
            self.q_net = GNNQNetwork(state_layout, ag_idx, action_dim, hidden_dim,
                                     gnn_message_dim, gnn_hidden_dim).to(self.device)
            self.target_net = GNNQNetwork(state_layout, ag_idx, action_dim, hidden_dim,
                                          gnn_message_dim, gnn_hidden_dim).to(self.device)
        else:
            self.q_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return th.tensor(data, dtype=th.float32)
        else:
            return data

    def select_action(self, state, env):
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
            action = th.tensor([[secrets.randbelow(self.action_dim)]], dtype=th.long)

        return action

    def store_transition(self, state, action, next_state, done, reward):
        state = self.convert_to_tensor(state).to(self.device)
        next_state = self.convert_to_tensor(next_state).to(self.device)
        reward = self.convert_to_tensor(reward).to(self.device)
        action = self.convert_to_tensor(action).to(self.device)

        self.memory.push(state, action, next_state, done, reward)

    def soft_update_target_net(self):
        # In place: target = q * tau + target * (1 - tau), same arithmetic as a state_dict round trip
        with th.no_grad():
            for target, source in zip(self.target_net.parameters(), self.q_net.parameters()):
                target.copy_(source * self.tau + target * (1 - self.tau))
            for target, source in zip(self.target_net.buffers(), self.q_net.buffers()):
                target.copy_(source * self.tau + target * (1 - self.tau))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = th.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=th.bool)

        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if non_final_next_states_list:
            non_final_next_states = th.cat(non_final_next_states_list)
        else:
            non_final_next_states = th.empty((0, self.state_dim))

        state_batch = th.cat([s.to(self.device) for s in batch.state], dim=0)
        action_batch = th.cat([a.to(self.device).long() for a in batch.action], dim=0)
        reward_batch = th.cat([r.to(self.device).float() for r in batch.reward], dim=0)
        done_batch = th.tensor(batch.done, device=self.device, dtype=th.bool)

        # Enforce constraint on current Q-values
        if self.force_nt_when_empty:
            all_agent_queues = state_batch[:, -self.num_agents:]
            current_queue = all_agent_queues[:, self.ag_idx]
            queue_empty_mask = (current_queue == 0.0)
            nt_action_idx = self.action_dim - 1
            corrected_action_batch = th.where(
                queue_empty_mask.unsqueeze(1),
                th.full_like(action_batch, nt_action_idx),
                action_batch
            )
            state_action_values = self.q_net(state_batch).gather(1, corrected_action_batch)
        else:
            state_action_values = self.q_net(state_batch).gather(1, action_batch)

        next_state_values = th.zeros(self.batch_size, device=self.device)

        with th.no_grad():
            if non_final_next_states.shape[0] > 0:
                all_next_q_values = self.target_net(non_final_next_states)

                if self.force_nt_when_empty:
                    all_agent_queues = non_final_next_states[:, -self.num_agents:]
                    next_queue_value = all_agent_queues[:, self.ag_idx]
                    queue_empty_mask = (next_queue_value == 0.00)
                    nt_action_idx = self.action_dim - 1
                    best_actions = all_next_q_values.max(1).indices
                    forced_actions = th.where(
                        queue_empty_mask,
                        th.full_like(best_actions, nt_action_idx),
                        best_actions
                    )
                    next_q_values = all_next_q_values.gather(1, forced_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q_values = all_next_q_values.max(1).values

                next_state_values[non_final_mask] = next_q_values

            next_state_values = next_state_values.unsqueeze(1)

        # Compute expected state action values
        expected_state_action_values = th.where(
            done_batch.unsqueeze(1),
            reward_batch,
            reward_batch + self.gamma * next_state_values
        )

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values.float(), expected_state_action_values.float())

        if self.is_hysteretic_q:
            td_error = expected_state_action_values - state_action_values
            positive_td_mask = td_error > 0
            hysteretic_lr = th.where(positive_td_mask, self.hysteretic_high_lr, self.hysteretic_low_lr).view(-1, 1)
            criterion = nn.MSELoss(reduction='none')
            loss_per_sample = criterion(state_action_values, expected_state_action_values).view(-1, 1)
            scaled_loss_per_sample = hysteretic_lr * loss_per_sample
            loss = scaled_loss_per_sample.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

    def get_action_values(self, state):
        state = self.convert_to_tensor(state).to(self.device)

        with th.no_grad():
            action_values = self.q_net(state)

        return action_values