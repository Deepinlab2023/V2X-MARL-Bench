import csv
import numpy as np
import torch as th
from typing import Optional, Tuple, Union

from Environment.environment_utility import (
    sample_veh_position_from_timestep,
    random_sample,
)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class PPOHelper:
    """
    Shared helper for MAPPO and IPPO.

    This consolidates:
      - MAPPO Helper + BatchProcessing + ValueNormalizer
      - IPPO Helper + BatchProcessing + ValueNormalizer
      - Trainer-shared utilities (CSV naming, veh sampling, test finalize, returns norm)

    Notes:
      - MAPPO-specific feature-pruning utilities are included (overlap indices + fp-state creation).
      - GAE variants:
          * compute_gae_single: scalar advantages/returns (MAPPO centralized critic case)
          * compute_gae_agent_specific: per-agent advantages with scalar returns (MAPPO FP case)
          * compute_gae_ippo: per-agent advantages with scalar returns (IPPO case)
    """

    # -------------------------
    # File naming / logging
    # -------------------------
    @staticmethod
    def init_csv_logging(params, algo_name: str):
        env = params.env
        task_type = params.task_type
        n_agent = params.n_agent
        n_sc = params.n_sc
        ff_on = getattr(params, "fast_fading_enabled", getattr(env, "fast_fading_enabled", False))

        if task_type == "NFIG":
            loc_tag = "none" if params.loc is None else int(params.loc)
            csv_name = f"{algo_name}_trial_{params.trial_run}_NFIG{int(n_agent)}{int(n_sc)}_{loc_tag}.csv"

        elif task_type == "SIG":
            if params.loc is None:
                csv_name = f"{algo_name}_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"{algo_name}_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"

        elif task_type == "POSIG":
            if params.loc is None:
                csv_name = f"{algo_name}_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"{algo_name}_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"

        else:
            csv_name = f"{algo_name}_trial_{params.trial_run}_{task_type}.csv"

        csv_file = open(csv_name, "a", newline="")
        csv_writer = csv.writer(csv_file)
        return csv_file, csv_writer

    # -------------------------
    # Episode sampling
    # -------------------------
    @staticmethod
    def sample_episode_data(params):
        task_type = params.task_type

        if task_type == "NFIG" and params.loc is not None:
            return sample_veh_position_from_timestep(params.train_data, params.loc)

        if task_type in ("SIG", "POSIG") and params.loc is None:
            return random_sample(1, params.train_data)

        if task_type in ("SIG", "POSIG") and params.loc is not None:
            return sample_veh_position_from_timestep(params.train_data, params.loc)

        return params.train_data

    # -------------------------
    # PPO objective pieces
    # -------------------------
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
                clip_param,
            )

            loss_unclipped = (normalized_values - returns).pow(2)
            loss_clipped = (value_clip - returns).pow(2)

        else:
            value_clip = old_values + th.clamp(values - old_values, -clip_param, clip_param)
            loss_unclipped = (values - returns).pow(2)
            loss_clipped = (value_clip - returns).pow(2)

        return th.max(loss_unclipped, loss_clipped).mean()

    # -------------------------
    # GAE variants
    # -------------------------
    @staticmethod
    def compute_gae_single(rewards, values, dones, gamma, lam):
        """
        MAPPO centralized critic: one scalar value per timestep.
        Returns:
            returns: [T]
            advantages: [T]
        """
        T = len(rewards)

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
    def compute_gae_agent_specific(global_rewards, values, dones, gamma, lam):
        """
        MAPPO feature pruning: values are per-agent; returns remain scalar.
        Returns:
            returns: [T]
            advantages: [T, A]
        """
        T = len(global_rewards)
        n_agent = values[0].shape[0]

        d = th.tensor(dones, dtype=th.float32, device=device)

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

            delta = r_t + gamma * values[t + 1] * non_terminal - values[t]
            gae = delta + gamma * lam * non_terminal * gae
            advantages.insert(0, gae.clone())

            R = r_t + gamma * R * non_terminal
            returns.insert(0, R.clone())

        advantages = th.stack(advantages, dim=0)  # [T, A]
        returns = th.stack(returns, dim=0)        # [T]

        return returns, advantages

    @staticmethod
    def compute_gae_ippo(rewards, values, dones, gamma, lam):
        """
        IPPO: values are per-agent; returns are scalar.
        Returns:
            returns: [T]
            advantages: [T, A]
        """
        T = len(rewards)
        num_agents = len(values[0]) if hasattr(values[0], "__len__") else values[0].numel()

        advantages = [[0.0] * num_agents for _ in range(T)]
        returns = [0.0] * T

        values_list = []
        for v in values:
            if th.is_tensor(v):
                values_list.append(v.detach().cpu().tolist() if v.dim() > 0 else [v.item()])
            else:
                values_list.append(list(v))
        values_list.append([0.0] * num_agents)

        gae = [0.0] * num_agents
        R = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            r_t = float(rewards[t])

            R = r_t + gamma * R * mask
            returns[t] = R

            for a in range(num_agents):
                delta = r_t + gamma * values_list[t + 1][a] * mask - values_list[t][a]
                gae[a] = delta + gamma * lam * mask * gae[a]
                advantages[t][a] = gae[a]

        returns = th.tensor(returns, dtype=th.float32, device=device)
        advantages = th.tensor(advantages, dtype=th.float32, device=device)
        return returns, advantages

    # -------------------------
    # Returns normalization
    # -------------------------
    @staticmethod
    def normalize_returns(batch_returns, popart: bool, value_normalizer):
        if popart and value_normalizer is not None:
            value_normalizer.update(batch_returns.view(-1))
            batch_returns = value_normalizer.normalize(batch_returns)
        else:
            rtrn_mean = batch_returns.mean()
            rtrn_std = batch_returns.std(unbiased=False)
            batch_returns = (batch_returns - rtrn_mean) / rtrn_std.clamp_min(1e-8)
        return batch_returns

    # -------------------------
    # MAPPO feature pruning utilities
    # -------------------------
    @staticmethod
    def find_overlapping_indices(global_state,
                                 observation,
                                 agent_idx: int,
                                 timesteps: int,
                                 n_agent: int,
                                 subchannels: int,
                                 n_neighbor: int = 1,
                                 dest_idx: int = 0):
        A = n_agent
        M = subchannels

        g_len = int(global_state.numel())
        o_len = int(observation.numel())

        T = o_len - M - 3
        if T <= 0:
            raise ValueError(
                f"Cannot determine valid T from observation length: {o_len}. "
                f"With M={M}: expected o_len > {M + 3}"
            )

        expected_g_len = T + A + A * (A - 1) + M + A * M + A + A * M + A
        if g_len != expected_g_len:
            T_norm = 1
            expected_g_len_norm = T_norm + A + A * (A - 1) + M + A * M + A + A * M + A
            if g_len == expected_g_len_norm:
                T = T_norm
            else:
                raise ValueError(
                    f"Global state length mismatch: got {g_len}, expected {expected_g_len} "
                    f"(with T={T}, A={A}, M={M}) or {expected_g_len_norm} (with T=1)"
                )

        t_start = 0
        gi_start = T
        gji_start = gi_start + A
        gm_start = gji_start + A * (A - 1)
        gbi_start = gm_start + M
        gib_start = gbi_start + A * M
        iprev_start = gib_start + A
        q_start = iprev_start + A * M

        overlapping = []
        overlapping.extend(range(t_start, t_start + T))
        overlapping.append(gi_start + agent_idx)
        overlapping.append(gib_start + agent_idx)
        overlapping.extend(range(iprev_start + agent_idx * M, iprev_start + (agent_idx + 1) * M))
        overlapping.append(q_start + agent_idx)

        overlapping = sorted(set(overlapping))

        if overlapping and max(overlapping) >= g_len:
            raise RuntimeError(
                f"Overlap index out of bounds: max={max(overlapping)}, g_len={g_len}"
            )

        return overlapping

    @staticmethod
    def create_fp_state(global_state, observation, agent_idx, agent_id, timesteps, n_agent, subchannels):
        g = global_state if global_state.dim() == 1 else global_state.view(-1)
        obs = observation if observation.dim() == 1 else observation.view(-1)

        overlapping_indices = PPOHelper.find_overlapping_indices(
            g, obs, agent_idx, timesteps, n_agent, subchannels
        )

        total_indices = list(range(g.numel()))
        non_overlapping_indices = [i for i in total_indices if i not in overlapping_indices]
        non_overlapping_global_state = g[non_overlapping_indices]

        fp_state = th.cat([obs, non_overlapping_global_state, agent_id.squeeze(0)], dim=-1)
        return fp_state


class PPOBatchProcessing:
    """
    Unified batch collation for MAPPO and IPPO.

    For MAPPO:
      - FO: buffer items are (global_state_hist, joint_action_hist, log_prob_hist, value_hist, returns, advantages)
      - POSIG non-FP: (global_state_hist, observation_hist, joint_action_hist, log_prob_hist, value_hist, returns, advantages)
      - POSIG FP: (global_state_hist(fp_states), observation_hist, joint_action_hist, log_prob_hist, value_hist(per-agent), returns, advantages(per-agent))

    For IPPO:
      - buffer items are (states_hist, joint_action_hist, log_prob_hist, value_hist, returns, advantages)
        where states_hist is obs_history for POSIG, otherwise global_state_history.
    """

    def collate_mappo_batch(self, buffer, task_type, feature_pruning=False):
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
                    values_tensor = th.stack(values).detach()
                else:
                    values_tensor = th.stack([v.squeeze() for v in values], dim=0).detach()
            else:
                global_state, joint_action, log_probs, values, rtrn, advantages = data
                values_tensor = th.stack([v.squeeze() for v in values], dim=0).detach()

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

        batch_global_states = th.cat(batch_global_states, dim=0)
        batch_joint_actions = th.cat(batch_joint_actions, dim=0)
        batch_log_probs = th.cat(batch_log_probs, dim=0)
        batch_values = th.cat(batch_values, dim=0)
        batch_returns = th.cat(batch_returns, dim=0)
        batch_advantages = th.cat(batch_advantages, dim=0)

        if task_type == "POSIG":
            batch_observations = th.cat(batch_observations, dim=0)
            return (
                batch_global_states,
                batch_observations,
                batch_joint_actions,
                batch_log_probs,
                batch_values,
                batch_returns,
                batch_advantages,
            )

        return (
            batch_global_states,
            batch_joint_actions,
            batch_log_probs,
            batch_values,
            batch_returns,
            batch_advantages,
        )

    def collate_ippo_batch(self, buffer, task_type):
        batch_states = []
        batch_joint_actions = []
        batch_log_probs = []
        batch_values = []
        batch_returns = []
        batch_advantages = []

        for data in buffer:
            states, joint_actions, log_probs, values, rtrn, advantages = data

            if task_type == "POSIG":
                states_tensor = th.stack(states, dim=0).to(device)
            else:
                states_tensor = th.stack(states, dim=0).to(device)

            if isinstance(joint_actions[0], th.Tensor):
                joint_actions_tensor = th.stack(joint_actions, dim=0).to(device)
            else:
                joint_actions_tensor = th.tensor(joint_actions, dtype=th.long, device=device)

            log_probs_tensor = th.stack(log_probs, dim=0).to(device)
            values_tensor = th.stack(values, dim=0).to(device)

            batch_states.append(states_tensor)
            batch_joint_actions.append(joint_actions_tensor)
            batch_log_probs.append(log_probs_tensor)
            batch_values.append(values_tensor)
            batch_returns.append(rtrn.to(device))
            batch_advantages.append(advantages.to(device))

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
            batch_advantages,
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