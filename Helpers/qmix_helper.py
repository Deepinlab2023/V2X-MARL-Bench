import argparse
import random
from collections import deque, namedtuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "done", "reward"))


def QMIX_network_init(env, algo_params, env_name):
    """Initialize QMIX/VDN mixer args from environment and algo_params."""
    parser = argparse.ArgumentParser(description="QMIX Network")
    mix_args, _ = parser.parse_known_args()

    # For POSIG: mixer uses concatenated local observations
    # For SIG/NFIG: mixer uses global state
    if env_name == "POSIG":
        mix_args.state_shape = env.local_state_dim * env.n_agent
    else:
        mix_args.state_shape = env.state_dim

    mix_args.hyper_hidden_dim = algo_params.hyper_hidden_dim
    mix_args.qmix_hidden_dim = algo_params.qmix_hidden_dim
    mix_args.n_agents = env.n_agent
    mix_args.two_hyper_layers = algo_params.two_hyper_layers

    return mix_args


class QMixNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim),
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim),
            )
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(args.state_shape, args.qmix_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.qmix_hidden_dim, 1),
        )

    def forward(self, q_values: th.Tensor, states: th.Tensor) -> th.Tensor:
        episode_num = q_values.size(0)

        q_values = q_values.view(-1, 1, self.args.n_agents)
        states = states.reshape(-1, self.args.state_shape).float()

        w1 = th.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)

        hidden = F.elu(th.bmm(q_values, w1) + b1)

        w2 = th.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = th.bmm(hidden, w2) + b2
        q_total = q_total.view(episode_num, -1, 1)
        return q_total


class VDNMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, q_values: th.Tensor, _states: th.Tensor) -> th.Tensor:
        episode_num = q_values.size(0)

        q_values = q_values.view(-1, 1, self.args.n_agents)
        w = th.ones(q_values.size(0), self.args.n_agents, 1, device=q_values.device)

        q_total = th.bmm(q_values, w)
        q_total = q_total.view(episode_num, -1, 1)
        return q_total


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# class QMIXLearner:
#     def __init__(
#         self,
#         agent_list,
#         device: th.device,
#         is_vdn: bool,
#         mix_args,
#         memory_capacity: int = 10000,
#         batch_size: int = 64,
#         agent_lr: float = 1e-5,
#         mixer_lr: float = 1e-6,
#     ):
#         self.memory = ReplayMemory(memory_capacity)
#         self.agent_list = list(agent_list)
#         self.batch_size = batch_size
#         self.device = device
#         self.is_vdn = is_vdn
#         self.agent_lr = agent_lr
#         self.mixer_lr = mixer_lr

class QMIXLearner:
    def __init__(
        self,
        agent_list,
        device: th.device,
        is_vdn: bool,
        mix_args,
        env_name: str,
        memory_capacity: int,
        batch_size: int,
        agent_lr: float,
        mixer_lr: float,
    ):

        self.memory = ReplayMemory(memory_capacity)
        self.agent_list = list(agent_list)
        self.batch_size = batch_size
        self.device = device
        self.is_vdn = is_vdn
        self.env_name = env_name
        self.agent_lr = agent_lr
        self.mixer_lr = mixer_lr

        self.optimizer = None

        mixer_cls = VDNMixer if is_vdn else QMixNet
        self.eval_mixing_net = mixer_cls(mix_args).to(self.device)
        self.target_mixing_net = mixer_cls(mix_args).to(self.device)
        self.target_mixing_net.load_state_dict(self.eval_mixing_net.state_dict())

    def add_agent(self, agent) -> None:
        self.agent_list.append(agent)

    def init_model(self) -> None:
        agent_parameters = []
        for agent in self.agent_list:
            agent_parameters.extend(list(agent.q_net.parameters()))

        mixer_parameters = list(self.eval_mixing_net.parameters())

        self.optimizer = th.optim.Adam(
            [
                {"params": agent_parameters, "lr": self.agent_lr},
                {"params": mixer_parameters, "lr": self.mixer_lr},
            ]
        )

    def store_transition(self, *args) -> None:
        self.memory.push(*args)

    def sample_batch(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}
        next_state_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}
        action_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}

        for agent_state, agent_next_state, agent_action in zip(batch.state, batch.next_state, batch.action):
            for ag_idx in range(len(self.agent_list)):
                if agent_state[ag_idx] is not None:
                    state_list[ag_idx].append(th.tensor(agent_state[ag_idx], dtype=th.float32))
                if agent_next_state[ag_idx] is not None:
                    next_state_list[ag_idx].append(th.tensor(agent_next_state[ag_idx], dtype=th.float32))
                if agent_action[ag_idx] is not None:
                    action_value = int(agent_action[ag_idx].item())
                    action_list[ag_idx].append(th.tensor([action_value], dtype=th.int64, device=self.device))

        reward_batch = th.tensor(np.vstack(batch.reward), device=self.device)
        done_batch = th.tensor(batch.done, device=self.device, dtype=th.bool)
        return state_list, next_state_list, action_list, reward_batch, done_batch

    def centralized_training(self):
        sample = self.sample_batch()
        if sample is None:
            return None

        state_list, next_state_list, action_list, reward_batch, done_batch = sample

        local_qs = []
        next_local_qs = []

        # Collect all agent state batches for concatenation (for mixer)
        all_state_batches = []
        all_next_state_batches = []

        for ag_idx, agent in enumerate(self.agent_list):
            state_batch = th.cat(state_list[ag_idx]).to(self.device)
            next_state_batch = th.cat(next_state_list[ag_idx]).to(self.device)
            action_batch = th.cat(action_list[ag_idx]).unsqueeze(1).to(self.device)

            all_state_batches.append(state_batch)
            all_next_state_batches.append(next_state_batch)

            if agent.force_nt_when_empty:
                all_agent_queues = state_batch[:, -agent.num_agents:]
                current_queue = all_agent_queues[:, agent.ag_idx]
                queue_empty_mask = current_queue == 0.0
                nt_action_idx = agent.action_dim - 1
                corrected_action_batch = th.where(
                    queue_empty_mask.unsqueeze(1),
                    th.full_like(action_batch, nt_action_idx),
                    action_batch,
                )
                state_action_values = agent.q_net(state_batch).gather(1, corrected_action_batch).view(-1, 1)
            else:
                state_action_values = agent.q_net(state_batch).gather(1, action_batch).view(-1, 1)

            local_qs.append(state_action_values)

            non_final_next_states = [s for s in next_state_list[ag_idx] if s is not None]
            if not non_final_next_states:
                continue

            non_final_next_states = th.cat(non_final_next_states).view(-1, agent.state_dim).to(self.device)

            with th.no_grad():
                all_next_q = agent.target_net(non_final_next_states)

                if agent.force_nt_when_empty:
                    all_agent_queues = non_final_next_states[:, -agent.num_agents:]
                    next_queue = all_agent_queues[:, agent.ag_idx]
                    queue_empty_mask = next_queue == 0.0
                    nt_action_idx = agent.action_dim - 1

                    best_actions = all_next_q.max(1).indices
                    forced_actions = th.where(
                        queue_empty_mask,
                        th.full_like(best_actions, nt_action_idx),
                        best_actions,
                    )
                    next_q = all_next_q.gather(1, forced_actions.unsqueeze(1)).squeeze(1).view(-1, 1)
                else:
                    next_q = all_next_q.max(1)[0].view(-1, 1)

                next_q[done_batch] = 0.0
                next_local_qs.append(next_q)

        local_qs = th.cat(local_qs, dim=1)
        next_local_qs = th.cat(next_local_qs, dim=1)

        # # Concatenate all agent states for mixer input
        # # This works for both POSIG (local obs) and SIG/NFIG (global state - all same)
        # concat_state_batch = th.cat(all_state_batches, dim=1)
        # concat_next_state_batch = th.cat(all_next_state_batches, dim=1)

        # For POSIG: concatenate local observations
        # For SIG/NFIG: use single global state (all agents have same state)
        if self.env_name == "POSIG":
            concat_state_batch = th.cat(all_state_batches, dim=1)
            concat_next_state_batch = th.cat(all_next_state_batches, dim=1)
        else:
            concat_state_batch = all_state_batches[0]
            concat_next_state_batch = all_next_state_batches[0]


        Q_tot = self.eval_mixing_net(local_qs, concat_state_batch)
        next_Q_tot = self.target_mixing_net(next_local_qs, concat_next_state_batch)

        reward_batch = reward_batch.view(-1, 1, 1)
        done_batch_3d = done_batch.view(-1, 1, 1)

        target_value = th.where(
            done_batch_3d,
            reward_batch,
            reward_batch + self.agent_list[0].gamma * next_Q_tot,
        )

        loss = nn.MSELoss()(Q_tot, target_value.float())

        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_value_(self.parameters_to_optimize(), 100)
        self.optimizer.step()

        return loss.item()

    def parameters_to_optimize(self):
        params = []
        for agent in self.agent_list:
            params.extend(agent.q_net.parameters())
        params.extend(self.eval_mixing_net.parameters())
        return params

    def soft_update_target_net(self) -> None:
        target_sd = self.target_mixing_net.state_dict()
        eval_sd = self.eval_mixing_net.state_dict()

        tau = self.agent_list[0].tau
        for k in eval_sd:
            target_sd[k] = eval_sd[k] * tau + target_sd[k] * (1 - tau)

        self.target_mixing_net.load_state_dict(target_sd)