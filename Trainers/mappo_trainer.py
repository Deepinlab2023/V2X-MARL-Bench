import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import csv

from Networks.Actors.mappo_actor import ActorPS
from Networks.Critics.mappo_critic import Critic
from Helpers.mappo_helper import Helper, BatchProcessing, ValueNormalizer
from Helpers.plotting_helper import plot_test_returns
from Benchmarkers.mappo_test import MAPPOtester

from Environment.environment_utility import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAPPO_TrainerPS:

    def train_MAPPO_ParameterSharing(self, params):
        env = params.env
        task_type = params.task_type  # "NFIG" / "SIG" / "POSIG"
        n_agent = params.n_agent
        n_sc = params.n_sc
        ff_on = getattr(params, "fast_fading_enabled", getattr(env, "fast_fading_enabled", False))

        # -------------------------
        # Logging (CSV)
        # -------------------------
        if task_type == "NFIG":
            loc_tag = "none" if params.loc is None else int(params.loc)
            csv_name = f"MAPPO_trial_{params.trial_run}_NFIG{int(n_agent)}{int(n_sc)}_{loc_tag}.csv"
        elif task_type == "SIG":
            if params.loc is None:
                csv_name = f"MAPPO_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"MAPPO_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"
        elif task_type == "POSIG":
            if params.loc is None:
                csv_name = f"MAPPO_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"MAPPO_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"
        else:
            csv_name = f"MAPPO_trial_{params.trial_run}_{task_type}.csv"

        csv_file = open(csv_name, "a", newline="")
        csv_writer = csv.writer(csv_file)

        # -------------------------
        # Networks (MAPPO: centralized critic)
        # -------------------------
        feature_pruning = getattr(params, "feature_pruning", False)

        if task_type == "POSIG" and feature_pruning:
            # Actor uses local observation
            actor_shared = ActorPS(params.observation_dim, params.action_dim, params.actor_hidden_dim, n_agent).to(device)
            # FP critic input: obs + non_overlapping_global + agent_id
            # = local_state_dim + (global_state_dim - overlap) + n_agent
            # where overlap ≈ local_state_dim, so fp_dim ≈ global_state_dim + n_agent
            # But exact calculation: obs_dim + (global - overlapping) + agent_id
            # Overlapping = T + 1 + 1 + M + 1 = local_state_dim
            # Non-overlapping = global_state_dim - local_state_dim
            # fp_state_dim = local_state_dim + (global_state_dim - local_state_dim) + n_agent = global_state_dim + n_agent
            fp_critic_dim = params.global_state_dim + n_agent
            centralized_critic = Critic(fp_critic_dim, params.critic_hidden_dim, params.value_dim).to(device)
        elif task_type == "POSIG" and not feature_pruning:
            actor_shared = ActorPS(params.observation_dim, params.action_dim, params.actor_hidden_dim, n_agent).to(device)
            centralized_critic = Critic(params.global_state_dim, params.critic_hidden_dim, params.value_dim).to(device)
        else:
            actor_shared = ActorPS(params.state_dim, params.action_dim, params.actor_hidden_dim, n_agent).to(device)
            centralized_critic = Critic(params.state_dim, params.critic_hidden_dim, params.value_dim).to(device)

        value_normalizer = None
        if params.popart:
            value_normalizer = ValueNormalizer(centralized_critic.fc3, params.critic_rescale)

        actor_optimizer = th.optim.Adam(actor_shared.parameters(), lr=params.alpha)
        critic_optimizer = th.optim.Adam(centralized_critic.parameters(), lr=params.beta)

        # -------------------------
        # Rewards tracking
        # -------------------------
        test_rewards = []
        episode = 0

        # -------- initial test before training --------
        test_reward = MAPPOtester.test_MAPPO(actor_shared, params)
        test_rewards.append(test_reward)
        csv_writer.writerow([test_reward])
        csv_file.flush()
        print(f"Training reward at episode {episode}: {test_reward:.2f}")

        # -------------------------
        # Main training iterations
        # -------------------------
        for it in range(params.num_training_iteration):
            buffer = []

            for _ in range(params.batch_size):

                if task_type == "POSIG":
                    observation_history = []
                global_state_history = []
                global_reward_history = []
                joint_action_history = []
                log_prob_history = []
                value_history = []
                done_history = []

                # -------- sample episode data --------
                if task_type == "NFIG" and params.loc is not None:
                    sampled_data = sample_veh_position_from_timestep(params.train_data, params.loc)
                elif task_type in ("SIG", "POSIG") and params.loc is None:
                    sampled_data = random_sample(1, params.train_data)
                elif task_type in ("SIG", "POSIG") and params.loc is not None:
                    sampled_data = sample_veh_position_from_timestep(params.train_data, params.loc)
                else:
                    sampled_data = params.train_data

                # -------- init env --------
                env.train_data = sampled_data
                env.new_random_game()

                # -------- rollout --------
                for t in range(int(env.n_step_per_episode)):

                    actions = []
                    old_log_probs = []

                    # env expects actions shape [n_agent, 1, 2] = (subchannel, power)
                    rra = np.zeros((n_agent, 1, 2), dtype=np.int32)

                    if task_type == "POSIG":
                        observations = []
                        if feature_pruning:
                            fp_states = []
                            values = []

                    # Global state for centralized critic
                    if task_type == "POSIG":
                        # POSIG: use get_global_state for critic, get_state for actor
                        global_state_np = env.get_global_state(t)
                    else:
                        # NFIG/SIG: get_state(0, t) returns global state
                        global_state_np = env.get_state(0, t)
                    global_state = th.tensor(global_state_np, dtype=th.float32).squeeze().to(device)

                    for a in range(n_agent):
                        with th.no_grad():
                            agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float().to(device)

                            if task_type == "POSIG":
                                obs_np = env.get_state(a, t)
                                obs = th.tensor(obs_np, dtype=th.float32).squeeze().to(device)
                                observations.append(obs)

                                if feature_pruning:
                                    fp_state = Helper.create_fp_state(
                                        global_state, obs, a, agent_id,
                                        int(env.n_step_per_episode), n_agent, n_sc
                                    )
                                    fp_states.append(fp_state)

                                logits = actor_shared(obs, agent_id)
                            else:
                                logits = actor_shared(global_state, agent_id)

                            # action sampling (with optional masking)
                            if params.action_masking:
                                q = env.queue.flatten()[a]
                                action, log_prob, _ = actor_shared.action_sampler(logits, q)
                            else:
                                action, log_prob, _ = actor_shared.action_sampler(logits)

                            action_id = int(action.item())
                            old_log_probs.append(log_prob)
                            actions.append(action_id)

                            sc, pw = env.map_action_to_rra(action_id, a)
                            rra[a, 0, 0] = sc
                            rra[a, 0, 1] = pw

                            # Per-agent value for feature pruning
                            if task_type == "POSIG" and feature_pruning:
                                v = centralized_critic(fp_state)
                                values.append(v.squeeze().detach())

                    # Centralized critic value (non-FP cases)
                    with th.no_grad():
                        if not (task_type == "POSIG" and feature_pruning):
                            value = centralized_critic(global_state).squeeze(-1).detach()

                    # Stack per-agent data
                    joint_action = actions
                    old_log_probs_stacked = th.stack(old_log_probs).detach().to(device)  # [n_agent]

                    global_reward, done = env.step(rra, t)

                    # Store history
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

                # -------- GAE + store episode --------
                if task_type == "POSIG" and feature_pruning:
                    returns, advantages = Helper.compute_GAE_AS(
                        global_reward_history, value_history, done_history, params.gamma, params.lam
                    )
                    buffer.append((global_state_history, observation_history, joint_action_history,
                                   log_prob_history, value_history, returns, advantages))
                elif task_type == "POSIG" and not feature_pruning:
                    returns, advantages = Helper.compute_GAE_single(
                        global_reward_history, value_history, done_history, params.gamma, params.lam
                    )
                    buffer.append((global_state_history, observation_history, joint_action_history,
                                   log_prob_history, value_history, returns, advantages))
                else:
                    returns, advantages = Helper.compute_GAE_single(
                        global_reward_history, value_history, done_history, params.gamma, params.lam
                    )
                    buffer.append((global_state_history, joint_action_history, log_prob_history,
                                   value_history, returns, advantages))

                episode += 1

                # -------- periodic test --------
                if episode % params.test_interval == 0 and episode < params.training_episodes:
                    test_reward = MAPPOtester.test_MAPPO(actor_shared, params)
                    test_rewards.append(test_reward)
                    csv_writer.writerow([test_reward])
                    csv_file.flush()

                    plot_test_returns(
                        test_rewards, title="Test Return Over Time MAPPO", figure_id=1, pause=1.0,
                    )
                    print(f"Training reward at episode {episode}: {test_reward:.2f}")

            # -------------------------
            # Batch collation
            # -------------------------
            batch_processing = BatchProcessing()

            if task_type == "POSIG":
                (batch_global_states, batch_observations, batch_joint_actions,
                 batch_log_probs, batch_values, batch_returns, batch_advantages) = \
                    batch_processing.collate_batch(buffer, task_type, feature_pruning)
            else:
                (batch_global_states, batch_joint_actions, batch_log_probs,
                 batch_values, batch_returns, batch_advantages) = \
                    batch_processing.collate_batch(buffer, task_type)

            # batch_global_states: [B, state_dim] or [B, n_agent, fp_state_dim] for FP
            # batch_observations: [B, n_agent, obs_dim] for POSIG
            # batch_joint_actions: [B, n_agent]
            # batch_log_probs: [B, n_agent]
            # batch_values: [B] or [B, n_agent] for FP
            # batch_returns: [B]
            # batch_advantages: [B] or [B, n_agent] for FP

            # Returns normalization
            if params.popart and value_normalizer is not None:
                value_normalizer.update(batch_returns.view(-1))
                batch_returns = value_normalizer.normalize(batch_returns)
            else:
                rtrn_mean = batch_returns.mean()
                rtrn_std = batch_returns.std(unbiased=False)
                batch_returns = (batch_returns - rtrn_mean) / rtrn_std.clamp_min(1e-8)

            # -------------------------
            # Minibatch PPO updates
            # -------------------------
            if task_type == "POSIG":
                dataset = th.utils.data.TensorDataset(
                    batch_global_states, batch_observations, batch_joint_actions,
                    batch_log_probs, batch_values, batch_returns, batch_advantages
                )
            else:
                dataset = th.utils.data.TensorDataset(
                    batch_global_states, batch_joint_actions,
                    batch_log_probs, batch_values, batch_returns, batch_advantages
                )

            mini_batch_size = max(1, len(dataset) // params.num_mini_batches)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

            for _ in range(params.epochs):
                for batch in dataloader:

                    if task_type == "POSIG":
                        (global_states_mb, observations_mb, joint_actions_mb,
                         log_probs_mb, values_mb, returns_mb, advantages_mb) = batch
                        observations_mb = observations_mb.to(device)
                    else:
                        (global_states_mb, joint_actions_mb, log_probs_mb,
                         values_mb, returns_mb, advantages_mb) = batch

                    global_states_mb = global_states_mb.to(device)
                    joint_actions_mb = joint_actions_mb.to(device)
                    log_probs_mb = log_probs_mb.to(device)
                    values_mb = values_mb.to(device)
                    returns_mb = returns_mb.to(device)
                    advantages_mb = advantages_mb.to(device)

                    # Advantage normalization (non-FP: global advantages)
                    if not (task_type == "POSIG" and feature_pruning):
                        adv_mean = advantages_mb.mean()
                        adv_std = advantages_mb.std(unbiased=False)
                        advantages_normalized = (advantages_mb - adv_mean) / adv_std.clamp_min(1e-8)

                    # Extract queues for action masking (for non-POSIG)
                    if task_type != "POSIG":
                        queues_mb = global_states_mb[:, -n_agent:]

                    # ========== Critic Update ==========
                    critic_optimizer.zero_grad()

                    if task_type == "POSIG" and feature_pruning:
                        total_critic_loss = 0.0
                        for a in range(n_agent):
                            values_pred = centralized_critic(global_states_mb[:, a, :]).squeeze(-1)
                            critic_loss = Helper.critic_loss_fn(
                                values_pred,
                                values_mb[:, a],
                                returns_mb,
                                params.eps_clip,
                                params.popart,
                                value_normalizer
                            )
                            total_critic_loss += critic_loss
                        total_critic_loss = total_critic_loss / n_agent
                    else:
                        values_pred = centralized_critic(global_states_mb).squeeze(-1)
                        old_values = values_mb.squeeze(-1) if values_mb.dim() > 1 else values_mb
                        total_critic_loss = Helper.critic_loss_fn(
                            values_pred,
                            old_values,
                            returns_mb,
                            params.eps_clip,
                            params.popart,
                            value_normalizer
                        )

                    total_critic_loss.backward()
                    critic_optimizer.step()

                    # ========== Actor Update ==========
                    actor_optimizer.zero_grad()
                    total_actor_loss = 0.0

                    for a in range(n_agent):
                        agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float()
                        agent_id = agent_id.unsqueeze(0).repeat(joint_actions_mb.size(0), 1).to(device)

                        if task_type == "POSIG":
                            actor_input = observations_mb[:, a, :]
                        else:
                            actor_input = global_states_mb

                        logits = actor_shared(actor_input, agent_id)

                        # Action masking
                        if params.action_masking:
                            if task_type == "POSIG":
                                queue = actor_input[:, -1]
                            else:
                                queue = queues_mb[:, a]
                            done_mask = (queue <= 0)
                            if done_mask.any():
                                logits = logits.clone()
                                logits[done_mask, :-1] = -1e8

                        dist = Categorical(logits=logits)
                        action = joint_actions_mb[:, a]
                        old_log_prob = log_probs_mb[:, a]
                        new_log_prob = dist.log_prob(action)

                        # Advantage for this agent
                        if task_type == "POSIG" and feature_pruning:
                            adv_agent = advantages_mb[:, a]
                            adv_mean = adv_agent.mean()
                            adv_std = adv_agent.std(unbiased=False)
                            adv_agent = (adv_agent - adv_mean) / adv_std.clamp_min(1e-8)
                        else:
                            adv_agent = advantages_normalized

                        actor_loss = Helper.actor_loss_fn(
                            new_log_prob,
                            old_log_prob,
                            adv_agent,
                            params.eps_clip
                        )

                        entropy = dist.entropy().mean()
                        actor_loss = actor_loss - params.entropy_coef * entropy

                        total_actor_loss += actor_loss

                    total_actor_loss = total_actor_loss / n_agent
                    total_actor_loss.backward()
                    actor_optimizer.step()

        csv_file.close()
        return [], test_rewards