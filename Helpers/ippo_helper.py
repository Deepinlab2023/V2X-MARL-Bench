import torch as th

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class Helper:

    @staticmethod
    def actor_loss_fn(log_probs, old_log_probs, advantages, clip_param):
        ratio = th.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        return -th.min(surr1, surr2).mean()

    @staticmethod
    def critic_loss_fn(values, old_values, returns, clip_param, popart, value_normalizer):
        if popart:
            sigma = value_normalizer.sigma + 1e-8
            mu = value_normalizer.mu

            normalized_values = (values - mu) / sigma
            normalized_old = (old_values - mu) / sigma

            value_clip = normalized_old + th.clamp(
                normalized_values - normalized_old,
                -clip_param,
                clip_param
            )

            loss_unclipped = (normalized_values - returns).pow(2)
            loss_clipped = (value_clip - returns).pow(2)

        else:
            value_clip = old_values + th.clamp(values - old_values, -clip_param, clip_param)
            loss_unclipped = (values - returns).pow(2)
            loss_clipped = (value_clip - returns).pow(2)

        return th.max(loss_unclipped, loss_clipped).mean()

    @staticmethod
    def compute_GAE(rewards, values, dones, gamma, lam):
        """
        Compute GAE advantages and returns.

        Args:
            rewards: list[float] of length T
            values: list[Tensor[A]] of length T, each tensor has shape [n_agent]
            dones: list[bool] of length T
            gamma: discount factor
            lam: GAE lambda

        Returns:
            returns: Tensor[T] - scalar return per timestep
            advantages: Tensor[T, A] - per-agent advantages
        """
        T = len(rewards)
        num_agents = len(values[0]) if hasattr(values[0], '__len__') else values[0].numel()

        advantages = [[0.0] * num_agents for _ in range(T)]
        returns = [0.0] * T

        # Convert values to list of lists for easier indexing
        values_list = []
        for v in values:
            if th.is_tensor(v):
                values_list.append(v.detach().cpu().tolist() if v.dim() > 0 else [v.item()])
            else:
                values_list.append(list(v))
        # Append terminal values (zeros)
        values_list.append([0.0] * num_agents)

        gae = [0.0] * num_agents
        R = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            r_t = float(rewards[t])  # ensure Python float

            # Scalar return recursion (shared across agents)
            R = r_t + gamma * R * mask
            returns[t] = R

            # Per-agent GAE
            for a in range(num_agents):
                delta = r_t + gamma * values_list[t + 1][a] * mask - values_list[t][a]
                gae[a] = delta + gamma * lam * mask * gae[a]
                advantages[t][a] = gae[a]

        returns = th.tensor(returns, dtype=th.float32, device=device)       # [T]
        advantages = th.tensor(advantages, dtype=th.float32, device=device) # [T, A]

        return returns, advantages


class BatchProcessing:

    def collate_batch(self, buffer, task_type):
        """
        Collate episodes in buffer into batched tensors.

        For POSIG: states are observations with shape [T, n_agent, obs_dim]
        For others: states are global states with shape [T, state_dim]

        Returns:
            batch_states: [total_T, n_agent, obs_dim] for POSIG, [total_T, state_dim] otherwise
            batch_joint_actions: [total_T, n_agent]
            batch_log_probs: [total_T, n_agent]
            batch_values: [total_T, n_agent]
            batch_returns: [total_T]
            batch_advantages: [total_T, n_agent]
        """
        batch_states = []
        batch_joint_actions = []
        batch_log_probs = []
        batch_values = []
        batch_returns = []
        batch_advantages = []

        for data in buffer:
            states, joint_actions, log_probs, values, rtrn, advantages = data

            # States: stack list of tensors
            if task_type == "POSIG":
                # states is list of [n_agent, obs_dim] tensors
                states_tensor = th.stack(states, dim=0).to(device)  # [T, n_agent, obs_dim]
            else:
                # states is list of [state_dim] tensors
                states_tensor = th.stack(states, dim=0).to(device)  # [T, state_dim]

            # Joint actions: list of [n_agent] tensors
            if isinstance(joint_actions[0], th.Tensor):
                joint_actions_tensor = th.stack(joint_actions, dim=0).to(device)  # [T, n_agent]
            else:
                joint_actions_tensor = th.tensor(joint_actions, dtype=th.long, device=device)

            # Log probs: list of [n_agent] tensors
            log_probs_tensor = th.stack(log_probs, dim=0).to(device)  # [T, n_agent]

            # Values: list of [n_agent] tensors
            values_tensor = th.stack(values, dim=0).to(device)  # [T, n_agent]

            batch_states.append(states_tensor)
            batch_joint_actions.append(joint_actions_tensor)
            batch_log_probs.append(log_probs_tensor)
            batch_values.append(values_tensor)
            batch_returns.append(rtrn.to(device))
            batch_advantages.append(advantages.to(device))

        # Concatenate across episodes
        batch_states = th.cat(batch_states, dim=0)
        batch_joint_actions = th.cat(batch_joint_actions, dim=0)
        batch_log_probs = th.cat(batch_log_probs, dim=0)
        batch_values = th.cat(batch_values, dim=0)
        batch_returns = th.cat(batch_returns, dim=0)
        batch_advantages = th.cat(batch_advantages, dim=0)

        return (
            batch_states,
            batch_joint_actions,
            batch_log_probs,
            batch_values,
            batch_returns,
            batch_advantages
        )


class ValueNormalizer:
    """
    Running mean/variance for value targets (returns).
    Optional PopArt head rescaling (output layer).
    """

    def __init__(self, output_layer, rescale=True):
        self.rescale = rescale
        self.out = output_layer

        self.mu = th.tensor(0.0, device=device)
        self.sigma = th.tensor(1.0, device=device)
        self.count = th.tensor(1e-4, device=device)

    @th.no_grad()
    def update(self, targets):
        batch_mean = targets.mean()
        batch_var = targets.var(unbiased=False)
        batch_cnt = th.tensor(float(targets.numel()), device=device)

        delta = batch_mean - self.mu
        new_cnt = self.count + batch_cnt
        new_mu = self.mu + delta * (batch_cnt / new_cnt)

        m_a = (self.sigma ** 2) * self.count
        m_b = batch_var * batch_cnt
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_cnt / new_cnt
        new_sigma = th.sqrt(M2 / new_cnt + 1e-8)

        if self.rescale:
            scale = self.sigma / new_sigma
            self.out.weight.mul_(scale)
            self.out.bias.mul_(scale)
            self.out.bias.add_((self.mu - new_mu) / new_sigma)

        self.mu = new_mu
        self.sigma = new_sigma
        self.count = new_cnt

    def normalize(self, x):
        return (x - self.mu) / (self.sigma + 1e-8)