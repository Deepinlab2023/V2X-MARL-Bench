import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
import csv

from Networks.Actors.ippo_actor import ActorPS
from Networks.Critics.ippo_critic import CriticPS
from Helpers.ippo_helper import Helper, BatchProcessing, ValueNormalizer
from Helpers.plotting_helper import plot_test_returns
from Benchmarkers.ippo_test import IPPOtester

from Environment.environment_utility import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class IPPO_TrainerPS:

    def train_IPPO_ParameterSharing(self, params):
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
            csv_name = f"IPPO_trial_{params.trial_run}_NFIG{int(n_agent)}{int(n_sc)}_{loc_tag}.csv"
        elif task_type == "SIG":
            if params.loc is None:
                csv_name = f"IPPO_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"IPPO_trial_{params.trial_run}_SIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"
        elif task_type == "POSIG":
            if params.loc is None:
                csv_name = f"IPPO_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_ML_{ff_on}.csv"
            else:
                csv_name = f"IPPO_trial_{params.trial_run}_POSIG{int(n_agent)}{int(n_sc)}_SL_{ff_on}_{params.loc}.csv"
        else:
            csv_name = f"IPPO_trial_{params.trial_run}_{task_type}.csv"

        csv_file = open(csv_name, "a", newline="")
        csv_writer = csv.writer(csv_file)

        # -------------------------
        # Networks
        # -------------------------
        if task_type == "POSIG":
            actor_shared = ActorPS(params.observation_dim, params.action_dim, params.actor_hidden_dim, n_agent).to(device)
            critic_shared = CriticPS(params.observation_dim, params.critic_hidden_dim, params.value_dim, n_agent).to(device)
        else:
            actor_shared = ActorPS(params.state_dim, params.action_dim, params.actor_hidden_dim, n_agent).to(device)
            critic_shared = CriticPS(params.state_dim, params.critic_hidden_dim, params.value_dim, n_agent).to(device)

        value_normalizer = None
        if params.popart:
            value_normalizer = ValueNormalizer(critic_shared.fc3, params.critic_rescale)

        actor_optimizer = th.optim.Adam(actor_shared.parameters(), lr=params.alpha)
        critic_optimizer = th.optim.Adam(critic_shared.parameters(), lr=params.beta)

        # -------------------------
        # Rewards tracking
        # -------------------------
        test_rewards = []
        episode = 0

        # -------- initial test before training --------
        test_reward = IPPOtester.test_IPPO(actor_shared, params)
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
                else:
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
                    values = []

                    # env expects actions shape [n_agent, 1, 2] = (subchannel, power)
                    rra = np.zeros((n_agent, 1, 2), dtype=np.int32)

                    if task_type == "POSIG":
                        observations = []
                    else:
                        # global state (shared across agents)
                        global_state_np = env.get_state(0, t)
                        global_state = th.tensor(global_state_np, dtype=th.float32).squeeze().to(device)

                    for a in range(n_agent):
                        with th.no_grad():
                            agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float().to(device)

                            if task_type == "POSIG":
                                obs_np = env.get_state(a, t)
                                obs = th.tensor(obs_np, dtype=th.float32).squeeze().to(device)
                                observations.append(obs)
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

                            # critic value
                            if task_type == "POSIG":
                                v = critic_shared(obs, agent_id)
                            else:
                                v = critic_shared(global_state, agent_id)
                            values.append(v.squeeze().detach())

                    # Stack per-agent data into tensors
                    joint_action = th.tensor(actions, dtype=th.long, device=device)       # [n_agent]
                    old_log_probs = th.stack(old_log_probs).detach().to(device)           # [n_agent]
                    values = th.stack(values).detach().to(device)                         # [n_agent]

                    global_reward, done = env.step(rra, t)

                    if task_type == "POSIG":
                        observation_history.append(th.stack(observations, dim=0))  # [n_agent, obs_dim]
                    else:
                        global_state_history.append(global_state)  # [state_dim]

                    joint_action_history.append(joint_action)
                    log_prob_history.append(old_log_probs)
                    global_reward_history.append(global_reward)
                    value_history.append(values)
                    done_history.append(done)

                    if done:
                        break

                # -------- GAE + store episode --------
                returns, advantages = Helper.compute_GAE(
                    global_reward_history, value_history, done_history, params.gamma, params.lam
                )

                episode += 1

                if task_type == "POSIG":
                    buffer.append((observation_history, joint_action_history, log_prob_history,
                                   value_history, returns, advantages))
                else:
                    buffer.append((global_state_history, joint_action_history, log_prob_history,
                                   value_history, returns, advantages))

                # -------- periodic test --------
                if episode % params.test_interval == 0 and episode < params.training_episodes:
                    test_reward = IPPOtester.test_IPPO(actor_shared, params)
                    test_rewards.append(test_reward)
                    csv_writer.writerow([test_reward])
                    csv_file.flush()

                    plot_test_returns(
                        test_rewards, title="Test Return Over Time IPPO", figure_id=1, pause=1.0,
                    )
                    print(f"Training reward at episode {episode}: {test_reward:.2f}")

            # -------------------------
            # Batch collation
            # -------------------------
            batch_processing = BatchProcessing()

            (batch_states, batch_joint_actions, batch_log_probs,
             batch_values, batch_returns, batch_advantages) = batch_processing.collate_batch(buffer, task_type)

            # batch_states: [B, n_agent, obs_dim] for POSIG, [B, state_dim] for others
            # batch_joint_actions: [B, n_agent]
            # batch_log_probs: [B, n_agent]
            # batch_values: [B, n_agent]
            # batch_returns: [B]
            # batch_advantages: [B, n_agent]

            # Returns normalization
            if params.popart and value_normalizer is not None:
                value_normalizer.update(batch_returns)
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
                    batch_states, batch_joint_actions, batch_log_probs,
                    batch_values, batch_returns, batch_advantages
                )
            else:
                dataset = th.utils.data.TensorDataset(
                    batch_states, batch_joint_actions, batch_log_probs,
                    batch_values, batch_returns, batch_advantages
                )

            mini_batch_size = max(1, len(dataset) // params.num_mini_batches)
            dataloader = th.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

            for _ in range(params.epochs):
                for batch in dataloader:

                    if task_type == "POSIG":
                        observations_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = batch
                        observations_mb = observations_mb.to(device)  # [mb, n_agent, obs_dim]
                    else:
                        global_states_mb, joint_actions_mb, log_probs_mb, values_mb, returns_mb, advantages_mb = batch
                        global_states_mb = global_states_mb.to(device)  # [mb, state_dim]

                    joint_actions_mb = joint_actions_mb.to(device)   # [mb, n_agent]
                    log_probs_mb = log_probs_mb.to(device)           # [mb, n_agent]
                    values_mb = values_mb.to(device)                 # [mb, n_agent]
                    returns_mb = returns_mb.to(device)               # [mb]
                    advantages_mb = advantages_mb.to(device)         # [mb, n_agent]

                    # Advantage normalization (per-agent)
                    adv_mean = advantages_mb.mean(dim=0, keepdim=True)
                    adv_std = advantages_mb.std(dim=0, unbiased=False, keepdim=True)
                    advantages_normalized = (advantages_mb - adv_mean) / adv_std.clamp_min(1e-8)

                    # Extract queues for action masking (for non-POSIG)
                    if task_type != "POSIG":
                        queues_mb = global_states_mb[:, -n_agent:]  # [mb, n_agent]

                    # ========== Critic Update ==========
                    critic_optimizer.zero_grad()
                    total_critic_loss = 0.0

                    for a in range(n_agent):
                        agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float()
                        agent_id = agent_id.unsqueeze(0).repeat(joint_actions_mb.size(0), 1).to(device)

                        if task_type == "POSIG":
                            critic_input = observations_mb[:, a, :]  # [mb, obs_dim]
                        else:
                            critic_input = global_states_mb  # [mb, state_dim]

                        values_pred = critic_shared(critic_input, agent_id).squeeze(-1)  # [mb]

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
                    total_critic_loss.backward()
                    critic_optimizer.step()

                    # ========== Actor Update ==========
                    actor_optimizer.zero_grad()
                    total_actor_loss = 0.0

                    for a in range(n_agent):
                        agent_id = F.one_hot(th.tensor(a), num_classes=n_agent).float()
                        agent_id = agent_id.unsqueeze(0).repeat(joint_actions_mb.size(0), 1).to(device)

                        if task_type == "POSIG":
                            actor_input = observations_mb[:, a, :]  # [mb, obs_dim]
                        else:
                            actor_input = global_states_mb  # [mb, state_dim]

                        logits = actor_shared(actor_input, agent_id)

                        # Action masking
                        if params.action_masking:
                            if task_type == "POSIG":
                                queue = actor_input[:, -1]  # last element is queue
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

                        actor_loss = Helper.actor_loss_fn(
                            new_log_prob,
                            old_log_prob,
                            advantages_normalized[:, a],
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