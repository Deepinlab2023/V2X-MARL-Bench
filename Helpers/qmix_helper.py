import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
import random
import argparse

device = th.device("cuda" if th.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'done', 'reward'))


def QMIX_network_init(env):
    parser = argparse.ArgumentParser(description='QMIX Network')
    mix_args, unknown = parser.parse_known_args()

    mix_args.state_shape = env.stateDim
    # mix_args.state_shape = env.local_stateDim

    mix_args.hyper_hidden_dim = 128
    mix_args.qmix_hidden_dim = 32
    mix_args.n_agents = env.n_agent
    mix_args.two_hyper_layers = True

    
    return mix_args


class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        # 因为生成的hyper_w1需要是一个矩阵，而pytorch神经网络只能输出一个向量，
        # 所以就先输出长度为需要的 矩阵行*矩阵列 的向量，然后再转化成矩阵

        # args.n_agents是使用hyper_w1作为参数的网络的输入维度，args.qmix_hidden_dim是网络隐藏层参数个数
        # 从而经过hyper_w1得到(经验条数，args.n_agents * args.qmix_hidden_dim)的矩阵
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Sequential(nn.Linear(args.state_shape, args.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            # 经过hyper_w2得到(经验条数, 1)的矩阵
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        # hyper_w1得到的(经验条数，args.qmix_hidden_dim)矩阵需要同样维度的hyper_b1
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        # hyper_w2得到的(经验条数，1)的矩阵需要同样维度的hyper_b1
        self.hyper_b2 =nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(args.qmix_hidden_dim, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        # 传入的q_values是三维的，shape为(episode_num, max_episode_len， n_agents)

        # print("q_values: ", q_values.shape)
        # print("states: ", states.shape)

        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents) = (1920,1,5)
        states = states.reshape(-1, self.args.state_shape)  # (episode_num * max_episode_len, state_shape)

        w1 = th.abs(self.hyper_w1(states.float()))  # (1920, 160)
        b1 = self.hyper_b1(states.float())  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        # print("shape q_values: ", q_values.shape, " w1: ", w1.shape)
        
        hidden = F.elu(th.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = th.abs(self.hyper_w2(states.float()))  # (1920, 32)
        b2 = self.hyper_b2(states.float())  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = th.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)

        # print("w1 b1, w2 b2: ")

        return q_total

class VDNMixer(nn.Module):
    def __init__(self, args):
        super(VDNMixer, self).__init__()
        self.args = args

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (1920, 1, n_agents)
        
        # Hardcode w2 to 1 and b2 to 0
        w = th.ones(q_values.size(0), self.args.n_agents, 1).to(q_values.device)  # (1920, n_agents, 1)


        # Sum Q-values directly (equivalent to VDN)
        q_total = th.bmm(q_values, w)  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)


        return q_total


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




class QMIXLearner():
    def __init__(self, agent_list, device, is_vdn, mix_args, memory_capacity=10000, batch_size=64):

        self.memory = ReplayMemory(memory_capacity)
        self.agent_list = agent_list
        self.batch_size = batch_size
        self.device = device
        self.is_vdn = is_vdn

        self.optimizer = None

        if is_vdn:
            self.eval_mixing_net = VDNMixer(mix_args).to(self.device)
            self.target_mixing_net = VDNMixer(mix_args).to(self.device)
        else:
            self.eval_mixing_net = QMixNet(mix_args).to(self.device)
            self.target_mixing_net = QMixNet(mix_args).to(self.device)


        self.target_mixing_net.load_state_dict(self.eval_mixing_net.state_dict())



    def add_agent(self, agent):
        self.agent_list.append(agent)


    def init_model(self):
        
        agent_parameters = []

        for ag_idx in range(len(self.agent_list)):
            agent = self.agent_list[ag_idx]
            agent_parameters.extend(list(agent.q_net.parameters()))

        mixer_parameters = list(self.eval_mixing_net.parameters())

        self.optimizer = th.optim.Adam([
            {'params': agent_parameters, 'lr': 1e-5},    # Agent parameters with lr=3e-5
            {'params': mixer_parameters, 'lr': 1e-6}     # Mixing network parameters with lr=1e-5
        ])

    
    def store_transition(self, *args):
        self.memory.push(*args)


    def sample_batch(self):
        
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))  # Transpose the batch

        """
        Process state, next state, and action batches
        """
        state_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}
        next_state_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}
        action_list = {ag_idx: [] for ag_idx in range(len(self.agent_list))}
        
        for agent_state, agent_next_state, agent_action in zip(batch.state, batch.next_state, batch.action):
            for ag_idx in range(len(self.agent_list)):

                # Process state
                if agent_state[ag_idx] is not None:
                    state_list[ag_idx].append(th.tensor(agent_state[ag_idx], dtype=th.float32))
                else:
                    state_list[ag_idx].append(th.empty((0, self.agent_list[0].state_dim), dtype=th.float32))
                
                # Process next state
                if agent_next_state[ag_idx] is not None:
                    next_state_list[ag_idx].append(th.tensor(agent_next_state[ag_idx], dtype=th.float32))
                else:
                    next_state_list[ag_idx].append(th.empty((0, self.agent_list[0].state_dim), dtype=th.float32))
                
                # Process action
                if agent_action[ag_idx] is not None:
                    action_value = int(agent_action[ag_idx].item())  # Extract scalar from NumPy array
                    action_list[ag_idx].append(th.tensor([action_value], dtype=th.int64, device=self.device))  # Convert to tensor

        # Rewards and done flags
        reward_batch = th.tensor(np.vstack(batch.reward)).to(self.device)
        done_batch = th.tensor(batch.done, device=self.device, dtype=th.bool)

        return state_list, next_state_list, action_list, reward_batch, done_batch




    def centralized_training(self):
        # 1. Sample from memory D
        if len(self.memory) > self.batch_size:
            state_list, next_state_list, action_list, reward_batch, done_batch = self.sample_batch()
        else:
            return

        # Initialize Q value lists
        local_Qs = []  # For current state Q-values
        next_local_Qs = []  # For next state Q-values

        for ag_idx in range(len(self.agent_list)):
            agent = self.agent_list[ag_idx]

            state_batch = th.cat(state_list[ag_idx]).to(self.device)
            next_state_batch = th.cat(next_state_list[ag_idx]).to(self.device)
            action_batch = th.cat(action_list[ag_idx]).unsqueeze(1).to(self.device)

            # ============= ADDED: Enforce constraint on current Q-values =============
            if agent.force_nt_when_empty:
                all_agent_queues = state_batch[:, -agent.num_agents:]
                current_queue = all_agent_queues[:, agent.ag_idx]
                queue_empty_mask = (current_queue == 0.0)
                nt_action_idx = agent.action_dim - 1
                corrected_action_batch = th.where(
                    queue_empty_mask.unsqueeze(1),
                    th.full_like(action_batch, nt_action_idx),
                    action_batch
                )
                state_action_values = agent.q_net(state_batch).gather(1, corrected_action_batch).view(-1, 1)
            else:
                state_action_values = agent.q_net(state_batch).gather(1, action_batch).view(-1, 1)
            # ==========================================================================
            
            local_Qs.append(state_action_values)

            # Compute next state Q-values using target network for non-terminal states
            non_final_mask = ~done_batch
            non_final_next_states = [s for s in next_state_list[ag_idx] if s is not None]

            if non_final_next_states:
                non_final_next_states = th.cat(non_final_next_states).view(-1, agent.state_dim).to(self.device)
                
                with th.no_grad():
                    all_next_q_values = agent.target_net(non_final_next_states)  # Shape: [batch_size, n_actions]
                    
                    if agent.force_nt_when_empty:
                        # print(agent.force_nt_when_empty)
                        # Extract THIS agent's queue from next state
                        all_agent_queues = non_final_next_states[:, -agent.num_agents:]
                        next_queue_value = all_agent_queues[:, agent.ag_idx]
                        
                        # Check if queue is empty
                        queue_empty_mask = (next_queue_value == 0.0)
                        
                        # NT action is the last action index
                        nt_action_idx = agent.action_dim - 1
                        
                        # Select actions: argmax for non-empty queue, NT for empty queue
                        best_actions = all_next_q_values.max(1).indices
                        forced_actions = th.where(
                            queue_empty_mask,
                            th.full_like(best_actions, nt_action_idx),
                            best_actions
                        )
                        
                        # Get Q-values for the selected actions
                        next_state_action_values = all_next_q_values.gather(
                            1, forced_actions.unsqueeze(1)
                        ).squeeze(1).view(-1, 1)  # Shape: [batch_size, 1]
                    else:
                        # Standard QMIX: just take max Q-value
                        next_state_action_values = all_next_q_values.max(1)[0].view(-1, 1)  # Max Q-values
                    
                    # Set Q-values to 0 for terminal states
                    next_state_action_values[done_batch] = 0
                    next_local_Qs.append(next_state_action_values)

        # Combine local Q-values for current states using the QMIX mixing network
        local_Qs = th.cat(local_Qs, dim=1)  # Shape: [batch_size, n_agents]
        next_local_Qs = th.cat(next_local_Qs, dim=1)  # Shape: [batch_size, n_agents]

        # Mix local Q-values into global Q_tot
        Q_tot = self.eval_mixing_net(local_Qs, state_batch)  # Shape: [batch_size, 1, 1]
        next_Q_tot = self.target_mixing_net(next_local_Qs, next_state_batch)  # Shape: [batch_size, 1, 1]

        # Reshape reward and done batches
        reward_batch = reward_batch.view(-1, 1, 1)  # [batch_size, 1, 1]
        done_batch = done_batch.view(-1, 1, 1)  # [batch_size, 1, 1]

        # Compute target value using Bellman equation
        target_value = th.where(
            done_batch,  # If done == True
            reward_batch,  # Use only reward_batch
            reward_batch + self.agent_list[0].gamma * next_Q_tot  # Apply Bellman equation if not done
        )

        # Compute loss (TD error)
        criterion = nn.MSELoss()
        loss = criterion(Q_tot, target_value.float())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_value_(self.parameters_to_optimize(), 100)
        self.optimizer.step()

        return loss.item()


    def parameters_to_optimize(self):
        """Helper method to get all parameters for gradient clipping"""
        params = []
        for agent in self.agent_list:
            params.extend(agent.q_net.parameters())
        params.extend(self.eval_mixing_net.parameters())
        return params

    def soft_update_target_net(self):

        target_mixing_net_state_dict = self.target_mixing_net.state_dict()
        eval_mixing_net_state_dict = self.eval_mixing_net.state_dict()

        for key in eval_mixing_net_state_dict:
            target_mixing_net_state_dict[key] = eval_mixing_net_state_dict[key]* self.agent_list[0].tau + target_mixing_net_state_dict[key]*(1-self.agent_list[0].tau)
        self.target_mixing_net.load_state_dict(target_mixing_net_state_dict)



