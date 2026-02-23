import csv
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.ppo_actor import PPOSharedActor
from Networks.Critics.ppo_critic import PPOCentralizedCritic

from Helpers.ppo_helper import PPOHelper, PPOBatchProcessing, ValueNormalizer

from Helpers.plotting_helper import plot_test_returns
from Benchmarkers.mappo_test import MAPPOtester

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAPPO_TrainerPS:
    @staticmethod
    def train_MAPPO_ParameterSharing(params):
        trainer = MAPPO_TrainerPS(params)
        return trainer.train()

    # ==========================
    #   INIT / SETUP
    # ==========================
    def __init__(self, params):
        self.params = params
        self.env = params.env  # required by provided code

        self.task_type = params.task_type  # "NFIG" / "SIG" / "POSIG"
        self.n_agent = params.n_agent
        self.n_sc = params.n_sc
        self.ff_on = getattr(params, "fast_fading_enabled", getattr(self.env, "fast_fading_enabled", False))
        self.feature_pruning = getattr(params, "feature_pruning", False)

        self.csv_file, self.csv_writer = PPOHelper.init_csv_logging(params, algo_name="MAPPO")

        # Networks / optimizers / PopArt
        (
            self.actor_shared,
            self.centralized_critic,
            self.actor_optimizer,
            self.critic_optimizer,
            self.value_normalizer,
        ) = self._init_networks_and_optimizers()

        self.test_rewards = []
        self.episode = 0

        self._initial_test()

    # ==========================
    #   TRAIN
    # ==========================
    def train(self):
        p = self.params
        try:
            for _it in range(p.num_training_iteration):
                buffer = []

                for _ in range(p.batch_size):
                    episode_tuple = self._collect_single_episode()
                    buffer.append(episode_tuple)

                    self.episode += 1
                    self._maybe_test()

                batch = self._collate_batch(buffer)
                batch = self._normalize_returns(batch)
                self._ppo_update_epochs(batch)

        finally:
            self.csv_file.close()

        return [], self.test_rewards

    # ==========================
    #   NETWORKS / OPTIMIZERS
    # ==========================
    def _init_networks_and_optimizers(self):
        p = self.params
        task_type = self.task_type
        n_agent = self.n_agent

        if task_type == "POSIG" and self.feature_pruning:
            actor_shared = PPOSharedActor(p.observation_dim, p.action_dim, p.actor_hidden_dim, n_agent).to(device)
            fp_critic_dim = p.global_state_dim + n_agent
            # centralized_critic = PPOCentralizedCritic(fp_critic_dim, p.critic_hidden_dim, p.value_dim).to(device)
            centralized_critic = PPOCentralizedCritic(fp_critic_dim, p).to(device)

        elif task_type == "POSIG" and (not self.feature_pruning):
            actor_shared = PPOSharedActor(p.observation_dim, p.action_dim, p.actor_hidden_dim, n_agent).to(device)
            # centralized_critic = PPOCentralizedCritic(p.global_state_dim, p.critic_hidden_dim, p.value_dim).to(device)
            centralized_critic = PPOCentralizedCritic(p.global_state_dim, p).to(device)

        else:
            actor_shared = PPOSharedActor(p.state_dim, p.action_dim, p.actor_hidden_dim, n_agent).to(device)
            # centralized_critic = PPOCentralizedCritic(p.state_dim, p.critic_hidden_dim, p.value_dim).to(device)
            centralized_critic = PPOCentralizedCritic(p.state_dim, p).to(device)

        value_normalizer = None
        if p.popart:
            value_normalizer = ValueNormalizer(centralized_critic.fc3, p.critic_rescale)

        actor_optimizer = th.optim.Adam(actor_shared.parameters(), lr=p.alpha)
        critic_optimizer = th.optim.Adam(centralized_critic.parameters(), lr=p.beta)

        return actor_shared, centralized_critic, actor_optimizer, critic_optimizer, value_normalizer

    # ==========================
    #   TESTING
    # ==========================
    def _initial_test(self):
        test_reward = MAPPOtester.test_MAPPO(self.actor_shared, self.params)
        self.test_rewards.append(test_reward)
        self.csv_writer.writerow([test_reward])
        self.csv_file.flush()
        print(f"Training reward at episode {self.episode}: {test_reward:.2f}")

    def _maybe_test(self):
        p = self.params
        if self.episode % p.test_interval == 0 and self.episode < p.training_episodes:
            test_reward = MAPPOtester.test_MAPPO(self.actor_shared, p)
            self.test_rewards.append(test_reward)
            self.csv_writer.writerow([test_reward])
            self.csv_file.flush()

            plot_test_returns(
                self.test_rewards,
                title="Test Return Over Time MAPPO",
                figure_id=1,
                pause=1.0,
            )
            print(f"Training reward at episode {self.episode}: {test_reward:.2f}")

    # ==========================
    #   EPISODE COLLECTION
    # ==========================
    def _get_global_state(self, t: int):
        env = self.env
        if self.task_type == "POSIG":
            global_state_np = env.get_global_state(t)
        else:
            global_state_np = env.get_state(0, t)
        return th.tensor(global_state_np, dtype=th.float32).squeeze().to(device)

    def _collect_single_episode(self):
        p = self.params
        env = self.env
        task_type = self.task_type
        n_agent = self.n_agent
        n_sc = self.n_sc
        feature_pruning = self.feature_pruning

        if task_type == "POSIG":
            observation_history = []
        global_state_history = []
        global_reward_history = []
        joint_action_history = []
        log_prob_history = []
        value_history = []
        done_history = []

        sampled_data = PPOHelper.sample_episode_data(p)

        env.train_data = sampled_data
        env.new_random_game()

        for t in range(int(env.n_step_per_episode)):
            actions = []
            old_log_probs = []

            rra = np.zeros((n_agent, 1, 2), dtype=np.int32)

            if task_type == "POSIG":
                observations = []
                if feature_pruning:
                    fp_states = []
                    values = []

            global_state = self._get_global_state(t)

            for a in range(n_agent):
                with th.no_grad():
                    agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float().to(device)

                    if task_type == "POSIG":
                        obs_np = env.get_state(a, t)
                        obs = th.tensor(obs_np, dtype=th.float32).squeeze().to(device)
                        observations.append(obs)

                        if feature_pruning:
                            fp_state = PPOHelper.create_fp_state(
                                global_state,
                                obs,
                                a,
                                agent_id,
                                int(env.n_step_per_episode),
                                n_agent,
                                n_sc,
                            )
                            fp_states.append(fp_state)

                        logits = self.actor_shared(obs, agent_id)
                    else:
                        logits = self.actor_shared(global_state, agent_id)

                    if p.action_masking:
                        q = env.queue.flatten()[a]
                        action, log_prob, _ = self.actor_shared.action_sampler(logits, q)
                    else:
                        action, log_prob, _ = self.actor_shared.action_sampler(logits)

                    action_id = int(action.item())
                    old_log_probs.append(log_prob)
                    actions.append(action_id)

                    sc, pw = env.map_action_to_rra(action_id, a)
                    rra[a, 0, 0] = sc
                    rra[a, 0, 1] = pw

                    if task_type == "POSIG" and feature_pruning:
                        v = self.centralized_critic(fp_state)
                        values.append(v.squeeze().detach())

            with th.no_grad():
                if not (task_type == "POSIG" and feature_pruning):
                    value = self.centralized_critic(global_state).squeeze(-1).detach()

            joint_action = actions
            old_log_probs_stacked = th.stack(old_log_probs).detach().to(device)

            global_reward, done = env.step(rra, t)

            if task_type == "POSIG" and feature_pruning:
                observations_stacked = th.stack(observations, dim=0).detach()
                values_stacked = th.stack(values, dim=0).detach()
                fp_states_stacked = th.stack(fp_states, dim=0).detach()

                observation_history.append(observations_stacked)
                global_state_history.append(fp_states_stacked)
                value_history.append(values_stacked)

            elif task_type == "POSIG" and not feature_pruning:
                observations_stacked = th.stack(observations, dim=0).detach()

                observation_history.append(observations_stacked)
                global_state_history.append(global_state)
                value_history.append(value)

            else:
                global_state_history.append(global_state)
                value_history.append(value)

            joint_action_history.append(joint_action)
            log_prob_history.append(old_log_probs_stacked)
            global_reward_history.append(float(global_reward[0, 0]))
            done_history.append(done)

            if done:
                break

        if task_type == "POSIG" and feature_pruning:
            returns, advantages = PPOHelper.compute_gae_agent_specific(
                global_reward_history, value_history, done_history, p.gamma, p.lam
            )
            return (
                global_state_history,
                observation_history,
                joint_action_history,
                log_prob_history,
                value_history,
                returns,
                advantages,
            )

        returns, advantages = PPOHelper.compute_gae_single(
            global_reward_history, value_history, done_history, p.gamma, p.lam
        )

        if task_type == "POSIG":
            return (
                global_state_history,
                observation_history,
                joint_action_history,
                log_prob_history,
                value_history,
                returns,
                advantages,
            )

        return (
            global_state_history,
            joint_action_history,
            log_prob_history,
            value_history,
            returns,
            advantages,
        )

    # ==========================
    #   BATCH COLLATION
    # ==========================
    def _collate_batch(self, buffer):
        batch_processing = PPOBatchProcessing()
        return batch_processing.collate_mappo_batch(buffer, self.task_type, self.feature_pruning)

    # ==========================
    #   RETURNS NORMALIZATION
    # ==========================
    def _normalize_returns(self, batch):
        p = self.params

        if self.task_type == "POSIG":
            (
                batch_global_states,
                batch_observations,
                batch_joint_actions,
                batch_log_probs,
                batch_values,
                batch_returns,
                batch_advantages,
            ) = batch
        else:
            (
                batch_global_states,
                batch_joint_actions,
                batch_log_probs,
                batch_values,
                batch_returns,
                batch_advantages,
            ) = batch

        batch_returns = PPOHelper.normalize_returns(batch_returns, p.popart, self.value_normalizer)

        if self.task_type == "POSIG":
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

    # ==========================
    #   PPO UPDATES
    # ==========================
    def _make_dataset(self, batch):
        if self.task_type == "POSIG":
            return th.utils.data.TensorDataset(*batch)

        return th.utils.data.TensorDataset(*batch)

    def _ppo_update_epochs(self, batch):
        p = self.params
        task_type = self.task_type
        n_agent = self.n_agent
        feature_pruning = self.feature_pruning

        dataset = self._make_dataset(batch)
        mini_batch_size = max(1, len(dataset) // p.num_mini_batches)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        for _ in range(p.epochs):
            for mb in dataloader:

                if task_type == "POSIG":
                    (
                        global_states_mb,
                        observations_mb,
                        joint_actions_mb,
                        log_probs_mb,
                        values_mb,
                        returns_mb,
                        advantages_mb,
                    ) = mb
                    observations_mb = observations_mb.to(device)
                else:
                    (
                        global_states_mb,
                        joint_actions_mb,
                        log_probs_mb,
                        values_mb,
                        returns_mb,
                        advantages_mb,
                    ) = mb

                global_states_mb = global_states_mb.to(device)
                joint_actions_mb = joint_actions_mb.to(device)
                log_probs_mb = log_probs_mb.to(device)
                values_mb = values_mb.to(device)
                returns_mb = returns_mb.to(device)
                advantages_mb = advantages_mb.to(device)

                if not (task_type == "POSIG" and feature_pruning):
                    adv_mean = advantages_mb.mean()
                    adv_std = advantages_mb.std(unbiased=False)
                    advantages_normalized = (advantages_mb - adv_mean) / adv_std.clamp_min(1e-8)

                if task_type != "POSIG":
                    queues_mb = global_states_mb[:, -n_agent:]

                # ---- Critic update ----
                self.critic_optimizer.zero_grad()

                if task_type == "POSIG" and feature_pruning:
                    total_critic_loss = 0.0
                    for a in range(n_agent):
                        values_pred = self.centralized_critic(global_states_mb[:, a, :]).squeeze(-1)
                        critic_loss = PPOHelper.critic_loss_fn(
                            values_pred,
                            values_mb[:, a],
                            returns_mb,
                            p.eps_clip,
                            p.popart,
                            self.value_normalizer,
                        )
                        total_critic_loss += critic_loss
                    total_critic_loss = total_critic_loss / n_agent
                else:
                    values_pred = self.centralized_critic(global_states_mb).squeeze(-1)
                    old_values = values_mb.squeeze(-1) if values_mb.dim() > 1 else values_mb
                    total_critic_loss = PPOHelper.critic_loss_fn(
                        values_pred,
                        old_values,
                        returns_mb,
                        p.eps_clip,
                        p.popart,
                        self.value_normalizer,
                    )

                total_critic_loss.backward()
                self.critic_optimizer.step()

                # ---- Actor update ----
                self.actor_optimizer.zero_grad()
                total_actor_loss = 0.0

                for a in range(n_agent):
                    agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float()
                    agent_id = agent_id.unsqueeze(0).repeat(joint_actions_mb.size(0), 1).to(device)

                    actor_input = observations_mb[:, a, :] if task_type == "POSIG" else global_states_mb
                    logits = self.actor_shared(actor_input, agent_id)

                    if p.action_masking:
                        queue = actor_input[:, -1] if task_type == "POSIG" else queues_mb[:, a]
                        done_mask = (queue <= 0)
                        if done_mask.any():
                            logits = logits.clone()
                            logits[done_mask, :-1] = -1e8

                    dist = Categorical(logits=logits)
                    action = joint_actions_mb[:, a]
                    old_log_prob = log_probs_mb[:, a]
                    new_log_prob = dist.log_prob(action)

                    if task_type == "POSIG" and feature_pruning:
                        adv_agent = advantages_mb[:, a]
                        adv_mean = adv_agent.mean()
                        adv_std = adv_agent.std(unbiased=False)
                        adv_agent = (adv_agent - adv_mean) / adv_std.clamp_min(1e-8)
                    else:
                        adv_agent = advantages_normalized

                    actor_loss = PPOHelper.actor_loss_fn(
                        new_log_prob,
                        old_log_prob,
                        adv_agent,
                        p.eps_clip,
                    )

                    entropy = dist.entropy().mean()
                    actor_loss = actor_loss - p.entropy_coef * entropy
                    total_actor_loss += actor_loss

                total_actor_loss = total_actor_loss / n_agent
                total_actor_loss.backward()
                self.actor_optimizer.step()
