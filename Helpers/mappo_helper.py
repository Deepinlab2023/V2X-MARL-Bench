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
    def compute_GAE_single(rewards, values, dones, gamma, lam):
        """
        Compute GAE for centralized critic (single value per timestep).

        Args:
            rewards: list[float] of length T
            values: list[Tensor] of length T, each tensor is scalar
            dones: list[bool] of length T
            gamma: discount factor
            lam: GAE lambda

        Returns:
            returns: Tensor[T] - scalar return per timestep
            advantages: Tensor[T] - scalar advantages
        """
        T = len(rewards)

        # Convert values to tensor
        v = th.stack([vv.squeeze() for vv in values], dim=0).to(device)  # [T]
        v = th.cat([v, th.zeros(1, dtype=th.float32, device=device)], dim=0)  # [T+1]

        r = th.tensor(rewards, dtype=th.float32, device=device)  # [T]
        d = th.tensor(dones, dtype=th.float32, device=device)    # [T]

        advantages = th.zeros(T, dtype=th.float32, device=device)
        returns = th.zeros(T, dtype=th.float32, device=device)

        gae = th.tensor(0.0, dtype=th.float32, device=device)
        R = th.tensor(0.0, dtype=th.float32, device=device)

        for t in reversed(range(T)):
            non_terminal = 1.0 - d[t]
            delta = r[t] + gamma * v[t + 1] * non_terminal - v[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages[t] = gae

            R = r[t] + gamma * R * non_terminal
            returns[t] = R

        return returns, advantages

    @staticmethod
    def compute_GAE_AS(global_rewards, values, dones, gamma, lam):
        """
        Compute GAE for agent-specific values (feature pruning case).

        Args:
            global_rewards: list[float] of length T (global rewards)
            values: list[Tensor[n_agent]] of length T (per-agent values)
            dones: list[bool] of length T
            gamma: discount factor
            lam: GAE lambda

        Returns:
            returns: Tensor[T] - scalar return per timestep
            advantages: Tensor[T, n_agent] - per-agent advantages
        """
        T = len(global_rewards)
        n_agent = values[0].shape[0]

        d = th.tensor(dones, dtype=th.float32, device=device)

        # Bootstrap value at T: per-agent zeros
        values = values + [th.zeros_like(values[0])]

        gae = th.zeros(n_agent, dtype=th.float32, device=device)
        R = th.tensor(0.0, dtype=th.float32, device=device)

        advantages = []
        returns = []

        for t in reversed(range(T)):
            r_t = global_rewards[t]
            if not th.is_tensor(r_t):
                r_t = th.tensor(r_t, dtype=th.float32, device=device)

            non_terminal = 1.0 - d[t]

            # Per-agent TD error
            delta = r_t + gamma * values[t + 1] * non_terminal - values[t]

            # Per-agent GAE
            gae = delta + gamma * lam * non_terminal * gae
            advantages.insert(0, gae.clone())

            # Scalar return recursion
            R = r_t + gamma * R * non_terminal
            returns.insert(0, R.clone())

        advantages = th.stack(advantages, dim=0)  # [T, n_agent]
        returns = th.stack(returns, dim=0)        # [T]

        return returns, advantages

    @staticmethod
    def find_overlapping_indices(global_state,
                                 observation,
                                 agent_idx: int,
                                 timesteps: int,
                                 n_agent: int,
                                 subchannels: int,
                                 n_neighbor: int = 1,
                                 dest_idx: int = 0):
        """
        Return indices in the GLOBAL state that correspond to info already present
        in the LOCAL observation of agent `agent_idx`.

        New Global State layout (_get_state_SIG):
            [t_enc, g_i, g_ji, g_m, g_bi, g_ib, i_prev, queue]
            Sizes: [T, A, A*(A-1), M, A*M, A, A*M, A]

        New POSIG Observation layout (_get_state_POSIG):
            [t_enc, G_i, G_iB, I_prev, queue]
            Sizes: [T, 1, 1, M, 1]

        This function auto-detects T from the observation length.
        """
        A = n_agent
        M = subchannels

        g_len = int(global_state.numel())
        o_len = int(observation.numel())

        # Auto-detect T from observation length
        # Observation: T + 1 + 1 + M + 1 = T + M + 3
        # So: T = o_len - M - 3
        T = o_len - M - 3

        if T <= 0:
            raise ValueError(
                f"Cannot determine valid T from observation length: {o_len}. "
                f"With M={M}: expected o_len > {M + 3}"
            )

        # Verify against global state length
        # Global: T + A + A*(A-1) + M + A*M + A + A*M + A
        #       = T + A + A^2 - A + M + A*M + A + A*M + A
        #       = T + 2*A + A^2 - A + M + 2*A*M
        #       = T + A + A^2 + M + 2*A*M
        #       = T + A*(1 + A + 2*M) + M
        expected_g_len = T + A + A * (A - 1) + M + A * M + A + A * M + A
        
        if g_len != expected_g_len:
            # Try with normalized timestep (T=1)
            T_norm = 1
            expected_g_len_norm = T_norm + A + A * (A - 1) + M + A * M + A + A * M + A
            if g_len == expected_g_len_norm:
                T = T_norm
            else:
                raise ValueError(
                    f"Global state length mismatch: got {g_len}, expected {expected_g_len} "
                    f"(with T={T}, A={A}, M={M}) or {expected_g_len_norm} (with T=1)"
                )

        # Compute global block starts
        # [t_enc, g_i, g_ji, g_m, g_bi, g_ib, i_prev, queue]
        t_start = 0
        gi_start = T
        gji_start = gi_start + A
        gm_start = gji_start + A * (A - 1)
        gbi_start = gm_start + M
        gib_start = gbi_start + A * M
        iprev_start = gib_start + A
        q_start = iprev_start + A * M

        # Overlap indices - what's in observation that's also in global state
        # Observation: [t_enc, G_i(agent), G_iB(agent), I_prev(agent,:), queue(agent)]
        overlapping = []

        # 1) t-block (all T timestep values)
        overlapping.extend(range(t_start, t_start + T))

        # 2) G_i for this agent
        overlapping.append(gi_start + agent_idx)

        # 3) G_iB for this agent
        overlapping.append(gib_start + agent_idx)

        # 4) I_prev for this agent (M subchannels)
        overlapping.extend(range(iprev_start + agent_idx * M, iprev_start + (agent_idx + 1) * M))

        # 5) queue for this agent
        overlapping.append(q_start + agent_idx)

        overlapping = sorted(set(overlapping))

        if overlapping and max(overlapping) >= g_len:
            raise RuntimeError(
                f"Overlap index out of bounds: max={max(overlapping)}, g_len={g_len}"
            )

        return overlapping

    @staticmethod
    def create_fp_state(global_state, observation, agent_idx, agent_id, timesteps, n_agent, subchannels):
        """Create feature-pruned state for centralized critic."""
        g = global_state if global_state.dim() == 1 else global_state.view(-1)
        obs = observation if observation.dim() == 1 else observation.view(-1)

        overlapping_indices = Helper.find_overlapping_indices(
            g, obs, agent_idx, timesteps, n_agent, subchannels
        )

        total_indices = list(range(g.numel()))
        non_overlapping_indices = [i for i in total_indices if i not in overlapping_indices]
        non_overlapping_global_state = g[non_overlapping_indices]

        fp_state = th.cat([obs, non_overlapping_global_state, agent_id.squeeze(0)], dim=-1)
        return fp_state


class BatchProcessing:

    def collate_batch(self, buffer, task_type, feature_pruning=False):
        """
        Collate episodes in buffer into batched tensors.

        For POSIG: includes observations
        For others: global states only

        Returns vary based on task_type.
        """
        if task_type == "POSIG":
            batch_observations = []
        batch_global_states = []
        batch_joint_actions = []
        batch_log_probs = []
        batch_values = []
        batch_returns = []
        batch_advantages = []

        for data in buffer:
            if task_type == "POSIG":
                global_state, observations, joint_action, log_probs, values, rtrn, advantages = data
                observations_tensor = th.stack(observations).detach()
                batch_observations.append(observations_tensor)

                if feature_pruning:
                    values_tensor = th.stack(values).detach()  # [T, n_agent]
                else:
                    values_tensor = th.stack([v.squeeze() for v in values], dim=0).detach()  # [T]
            else:
                global_state, joint_action, log_probs, values, rtrn, advantages = data
                values_tensor = th.stack([v.squeeze() for v in values], dim=0).detach()  # [T]

            global_state_tensor = th.stack(global_state).detach()
            joint_action_tensor = th.tensor(joint_action, dtype=th.long)
            log_prob_values = [[lp.item() for lp in lps] for lps in log_probs]
            log_prob_tensor = th.tensor(log_prob_values, dtype=th.float32)

            batch_global_states.append(global_state_tensor)
            batch_joint_actions.append(joint_action_tensor)
            batch_log_probs.append(log_prob_tensor)
            batch_values.append(values_tensor)
            batch_returns.append(rtrn)
            batch_advantages.append(advantages)

        # Concatenate across episodes
        batch_global_states = th.cat(batch_global_states, dim=0)
        batch_joint_actions = th.cat(batch_joint_actions, dim=0)
        batch_log_probs = th.cat(batch_log_probs, dim=0)
        batch_values = th.cat(batch_values, dim=0)
        batch_returns = th.cat(batch_returns, dim=0)
        batch_advantages = th.cat(batch_advantages, dim=0)

        if task_type == "POSIG":
            batch_observations = th.cat(batch_observations, dim=0)
            return (batch_global_states, batch_observations, batch_joint_actions,
                    batch_log_probs, batch_values, batch_returns, batch_advantages)
        else:
            return (batch_global_states, batch_joint_actions, batch_log_probs,
                    batch_values, batch_returns, batch_advantages)


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