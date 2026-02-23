import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.ppo_actor import PPOSharedActor
from Networks.Critics.ppo_critic import PPOSharedCritic

from Helpers.ppo_helper import PPOHelper, PPOBatchProcessing, ValueNormalizer

from Helpers.plotting_helper import plot_test_returns
from Benchmarkers.ippo_test import IPPOtester

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class IPPO_TrainerPS:
    @staticmethod
    def train_IPPO_ParameterSharing(params):
        trainer = IPPO_TrainerPS(params)
        return trainer.train()

    # ==========================
    #   INIT / SETUP
    # ==========================
    def __init__(self, params):
        self.params = params
        self.env = params.env

        self.task_type = params.task_type
        self.n_agent = params.n_agent
        self.n_sc = params.n_sc
        self.ff_on = getattr(params, "fast_fading_enabled", getattr(self.env, "fast_fading_enabled", False))

        self.csv_file, self.csv_writer = PPOHelper.init_csv_logging(params, algo_name="IPPO")

        (
            self.actor_shared,
            self.critic_shared,
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
        n_agent = self.n_agent

        if self.task_type == "POSIG":
            actor_shared = PPOSharedActor(p.observation_dim, p.action_dim, p.actor_hidden_dim, n_agent).to(device)
            critic_shared = PPOSharedCritic(p.observation_dim, p.critic_hidden_dim, p.value_dim, n_agent).to(device)
        else:
            actor_shared = PPOSharedActor(p.state_dim, p.action_dim, p.actor_hidden_dim, n_agent).to(device)
            critic_shared = PPOSharedCritic(p.state_dim, p.critic_hidden_dim, p.value_dim, n_agent).to(device)

        value_normalizer = None
        if p.popart:
            value_normalizer = ValueNormalizer(critic_shared.fc3, p.critic_rescale)

        actor_optimizer = th.optim.Adam(actor_shared.parameters(), lr=p.alpha)
        critic_optimizer = th.optim.Adam(critic_shared.parameters(), lr=p.beta)

        return actor_shared, critic_shared, actor_optimizer, critic_optimizer, value_normalizer

    # ==========================
    #   TESTING
    # ==========================
    def _initial_test(self):
        test_reward = IPPOtester.test_IPPO(self.actor_shared, self.params)
        self.test_rewards.append(test_reward)
        self.csv_writer.writerow([test_reward])
        self.csv_file.flush()
        print(f"Training reward at episode {self.episode}: {test_reward:.2f}")

    def _maybe_test(self):
        p = self.params
        if self.episode % p.test_interval == 0 and self.episode < p.training_episodes:
            test_reward = IPPOtester.test_IPPO(self.actor_shared, p)
            self.test_rewards.append(test_reward)
            self.csv_writer.writerow([test_reward])
            self.csv_file.flush()

            plot_test_returns(
                self.test_rewards,
                title="Test Return Over Time IPPO",
                figure_id=1,
                pause=1.0,
            )
            print(f"Training reward at episode {self.episode}: {test_reward:.2f}")

    # ==========================
    #   EPISODE COLLECTION
    # ==========================
    def _get_global_state(self, t: int):
        global_state_np = self.env.get_state(0, t)
        return th.tensor(global_state_np, dtype=th.float32).squeeze().to(device)

    def _collect_single_episode(self):
        p = self.params
        env = self.env
        task_type = self.task_type
        n_agent = self.n_agent

        if task_type == "POSIG":
            observation_history = []
            global_state_history = None
        else:
            global_state_history = []
            observation_history = None

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
            values = []

            rra = np.zeros((n_agent, 1, 2), dtype=np.int32)

            if task_type == "POSIG":
                observations = []
                global_state = None
            else:
                global_state = self._get_global_state(t)

            for a in range(n_agent):
                with th.no_grad():
                    agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float().to(device)

                    if task_type == "POSIG":
                        obs_np = env.get_state(a, t)
                        obs = th.tensor(obs_np, dtype=th.float32).squeeze().to(device)
                        observations.append(obs)
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

                    v = self.critic_shared(obs, agent_id) if task_type == "POSIG" else self.critic_shared(global_state, agent_id)
                    values.append(v.squeeze().detach())

            joint_action = th.tensor(actions, dtype=th.long, device=device)
            old_log_probs_t = th.stack(old_log_probs).detach().to(device)
            values_t = th.stack(values).detach().to(device)

            global_reward, done = env.step(rra, t)

            if task_type == "POSIG":
                observation_history.append(th.stack(observations, dim=0))
            else:
                global_state_history.append(global_state)

            joint_action_history.append(joint_action)
            log_prob_history.append(old_log_probs_t)
            global_reward_history.append(global_reward)
            value_history.append(values_t)
            done_history.append(done)

            if done:
                break

        returns, advantages = PPOHelper.compute_gae_ippo(
            global_reward_history, value_history, done_history, p.gamma, p.lam
        )

        if task_type == "POSIG":
            return (
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
        return batch_processing.collate_ippo_batch(buffer, self.task_type)

    # ==========================
    #   RETURNS NORMALIZATION
    # ==========================
    def _normalize_returns(self, batch):
        p = self.params
        (
            batch_states,
            batch_joint_actions,
            batch_log_probs,
            batch_values,
            batch_returns,
            batch_advantages,
        ) = batch

        batch_returns = PPOHelper.normalize_returns(batch_returns, p.popart, self.value_normalizer)

        return (
            batch_states,
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
        return th.utils.data.TensorDataset(*batch)

    def _unpack_minibatch(self, mb):
        if self.task_type == "POSIG":
            observations_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = mb
            observations_mb = observations_mb.to(device)
            global_states_mb = None
        else:
            global_states_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = mb
            global_states_mb = global_states_mb.to(device)
            observations_mb = None

        joint_actions_mb = joint_actions_mb.to(device)
        log_probs_mb = log_probs_mb.to(device)
        values_mb = values_mb.to(device)
        returns_mb = returns_mb.to(device)
        advantages_mb = advantages_mb.to(device)

        return observations_mb, global_states_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb

    def _compute_advantages_normalized(self, advantages_mb):
        adv_mean = advantages_mb.mean(dim=0, keepdim=True)
        adv_std = advantages_mb.std(dim=0, unbiased=False, keepdim=True)
        return (advantages_mb - adv_mean) / adv_std.clamp_min(1e-8)

    def _ppo_update_epochs(self, batch):
        p = self.params
        task_type = self.task_type
        n_agent = self.n_agent

        dataset = self._make_dataset(batch)
        mini_batch_size = max(1, len(dataset) // p.num_mini_batches)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        for _ in range(p.epochs):
            for mb in dataloader:
                observations_mb, global_states_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = (
                    self._unpack_minibatch(mb)
                )

                advantages_normalized = self._compute_advantages_normalized(advantages_mb)

                if task_type != "POSIG":
                    queues_mb = global_states_mb[:, -n_agent:]

                # ---- Critic update ----
                self.critic_optimizer.zero_grad()
                total_critic_loss = 0.0

                for a in range(n_agent):
                    agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float()
                    agent_id = agent_id.unsqueeze(0).repeat(joint_actions_mb.size(0), 1).to(device)

                    critic_input = observations_mb[:, a, :] if task_type == "POSIG" else global_states_mb
                    values_pred = self.critic_shared(critic_input, agent_id).squeeze(-1)

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

                    actor_loss = PPOHelper.actor_loss_fn(
                        new_log_prob,
                        old_log_prob,
                        advantages_normalized[:, a],
                        p.eps_clip,
                    )

                    entropy = dist.entropy().mean()
                    actor_loss = actor_loss - p.entropy_coef * entropy
                    total_actor_loss += actor_loss

                total_actor_loss = total_actor_loss / n_agent
                total_actor_loss.backward()
                self.actor_optimizer.step()