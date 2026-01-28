import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import secrets
from Helpers.FullyObservable.idql_helper import *
import random


class QNetworkWithTower(nn.Module):
    """
    Feature Tower Architecture for Queue-Aware DQN
    
    Separates queue processing into dedicated tower to guarantee
    strong gradient flow and prevent feature dilution.
    """
    def __init__(self, n_observations, n_actions, queue_index):
        super(QNetworkWithTower, self).__init__()
        self.queue_index = queue_index
        self.n_observations = n_observations
        
        # Queue feature tower (dedicated processing)
        self.queue_tower = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Main network for other features
        self.main_network = nn.Sequential(
            nn.Linear(n_observations - 1, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Fusion head (concatenate and produce Q-values)
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        """
        Forward pass with feature tower architecture.
        
        Args:
            x: State tensor of shape (batch_size, n_observations)
        
        Returns:
            Q-values of shape (batch_size, n_actions)
        """
        # Split input: queue vs other features
        queue_feature = x[:, self.queue_index:self.queue_index+1]  # Shape: (batch, 1)
        
        # Concatenate features before and after queue index
        if self.queue_index == 0:
            other_features = x[:, 1:]
        elif self.queue_index == self.n_observations - 1:
            other_features = x[:, :-1]
        else:
            other_features = th.cat([
                x[:, :self.queue_index],
                x[:, self.queue_index+1:]
            ], dim=1)  # Shape: (batch, n_observations-1)
        
        # Process through separate towers
        h_queue = self.queue_tower(queue_feature)      # (batch, 32)
        h_main = self.main_network(other_features)     # (batch, 128)
        
        # Fuse and produce Q-values
        h_combined = th.cat([h_queue, h_main], dim=1)  # (batch, 160)
        q_values = self.fusion_head(h_combined)        # (batch, n_actions)
        
        return q_values


class QNetwork(nn.Module):
    """
    Original simple Q-network architecture (kept for backward compatibility)
    """
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer_norm1 = nn.LayerNorm(128)
    
        self.layer2 = nn.Linear(128, 128)
        self.layer_norm2 = nn.LayerNorm(128)
    
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer_norm1(self.layer1(x)))
        x = F.relu(self.layer_norm2(self.layer2(x)))
        return self.layer3(x)


class DQNAgent:
    def __init__(self, ag_idx, num_agents, state_dim, action_dim, is_hysteretic_q, 
                 memory_capacity=10000, batch_size=64, gamma=0.9, tau=0.005, 
                 force_nt_when_empty=False, use_feature_tower=False):
        """
        DQN Agent with optional Feature Tower architecture.
        
        Args:
            ag_idx: Agent index
            num_agents: Total number of agents
            state_dim: Dimension of state space
            action_dim: Dimension of action space (NT action is last index)
            is_hysteretic_q: Whether to use hysteretic Q-learning
            memory_capacity: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update parameter for target network
            force_nt_when_empty: Whether to enforce no-transmit when queue empty
            use_feature_tower: Whether to use feature tower architecture (default: True)
        """
        self.ag_idx = ag_idx
        self.num_agents = num_agents
        
        self.state_dim = state_dim
        self.action_dim = action_dim  # int
        self.is_hysteretic_q = is_hysteretic_q
        
        # NT constraint configuration
        self.force_nt_when_empty = force_nt_when_empty
        self.use_feature_tower = use_feature_tower
        
        # Queue position in state vector: last num_agents elements are queues
        self.queue_index = state_dim - num_agents + ag_idx

        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.eps_threshold = 0
        self.n_episode = 0
        self.lr = 1e-5
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Initialize Q-networks with appropriate architecture
        if use_feature_tower:
            self.q_net = QNetworkWithTower(state_dim, action_dim, self.queue_index).to(self.device)
            self.target_net = QNetworkWithTower(state_dim, action_dim, self.queue_index).to(self.device)
        else:
            self.q_net = QNetwork(state_dim, action_dim).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def convert_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            return th.tensor(data, dtype=th.float32)
        else:
            return data

    def select_action(self, state, env):
        """
        Select action with hard constraint enforcement when queue is empty.
        
        Args:
            state: Current state
            env: Environment (to access queue values)
        
        Returns:
            action: Selected action as tensor
        """

        # forced_joint = [9, 0, 6, 3]
        # action = forced_joint[self.ag_idx]
        # return th.tensor([[action]], dtype=th.long)

        # Force NT when queue is empty (only applies to SIG/POSIG tasks)
        if self.force_nt_when_empty and env.queue[self.ag_idx][0] == 0:
            action = th.tensor([[self.action_dim - 1]], dtype=th.long)
            return action

        # Normal epsilon-greedy selection
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
        target_net_state_dict = self.target_net.state_dict()
        q_net_state_dict = self.q_net.state_dict()
        for key in q_net_state_dict:
            target_net_state_dict[key] = q_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

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
                    # Extract THIS agent's queue from next state
                    all_agent_queues = non_final_next_states[:, -self.num_agents:]
                    next_queue_value = all_agent_queues[:, self.ag_idx]
                    
                    # Check if queue is empty
                    queue_empty_mask = (next_queue_value == 0.00)
                    
                    # NT action is the last action
                    nt_action_idx = self.action_dim - 1
                    
                    # Select actions: argmax for non-empty queue, NT for empty queue
                    best_actions = all_next_q_values.max(1).indices
                    forced_actions = th.where(
                        queue_empty_mask,
                        th.full_like(best_actions, nt_action_idx),
                        best_actions
                    )
                    
                    # Get Q-values for the selected actions
                    next_q_values = all_next_q_values.gather(1, forced_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Standard DQN
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
            high_lr = 1.1
            low_lr = 0.2
            positive_td_mask = td_error > 0
            hysteretic_lr = th.where(positive_td_mask, high_lr, low_lr).view(-1, 1)
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