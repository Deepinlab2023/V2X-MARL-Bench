import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.a2c_actor import A2CSharedActor, A2CActorNS
from Networks.Critics.a2c_critic import A2CCentralizedCritic
from Helpers.a2c_helper import A2CHelper
from Benchmarkers.maa2c_test import MAA2Ctester

from Helpers.env_helper import EnvironHelper

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAA2CTrainer:
    @staticmethod
    def train_MAA2C(params):
        trainer = MAA2CTrainer(params)
        return trainer.train()

    def __init__(self, params):
        self.params = params
        self._compatibility_checks()
        self.csv_file, self.csv_writer = self._init_logging()
        (
            self.actor_shared,
            self.actors,
            self.critic,
            self.opt_actor_shared,
            self.opt_actors,
            self.opt_critic,
        ) = self._init_networks_and_optimizers()

        self.episode_rewards = []
        self.test_rewards = []
        self.episode = 0

    def train(self):
        return A2CHelper.train_loop(self, ctde=True)

    def _compatibility_checks(self):
        A2CHelper.basic_a2c_compat_checks(self.params, algo_name="MAA2C")

    def _init_logging(self):
        return A2CHelper.init_csv_logging(self.params, algo_name="MAA2C", posig_tag="FNN")

    def _init_networks_and_optimizers(self):
        p = self.params

        if p.no_sharing:
            actors = [A2CActorNS(p.state_dim, p.action_dim, p.actor_hidden_dim).to(device)
                      for _ in range(p.num_agents)]
            actor_shared = None
        else:
            if p.env_name == "POSIG":
                base_actor_dim = p.observation_dim
            else:
                base_actor_dim = p.state_dim

            actor_input_dim = base_actor_dim + p.num_agents
            if p.prev_action_input:  # allowed only if rnn=True
                actor_input_dim += p.action_dim

            actor_shared = A2CSharedActor(actor_input_dim, p).to(device)
            actors = None

        critic = A2CCentralizedCritic(p).to(device)

        if p.no_sharing:
            opt_actors = [th.optim.Adam(a.parameters(), lr=p.alpha) for a in actors]
            opt_actor_shared = None
        else:
            opt_actor_shared = th.optim.Adam(actor_shared.parameters(), lr=p.alpha)
            opt_actors = None

        opt_critic = th.optim.Adam(critic.parameters(), lr=p.beta)

        return actor_shared, actors, critic, opt_actor_shared, opt_actors, opt_critic

    # ----------------------------------------------------- #
    #  Rollout
    # ----------------------------------------------------- #
    def _run_single_episode(self):
        p = self.params
        total_rewards = 0.0
        buffer = []
        done = False

        # Only track RNN state for POSIG + PS + RNN
        if (not p.no_sharing) and p.env_name == "POSIG" and p.rnn:
            hidden_state = [
                th.zeros(1, 1, p.actor_hidden_dim, device=device)
                for _ in range(p.num_agents)
            ]
            if p.prev_action_input:
                prev_actions = [
                    th.zeros(1, p.action_dim, device=device)
                    for _ in range(p.num_agents)
                ]
            else:
                prev_actions = None
        else:
            hidden_state = None
            prev_actions = None

        num_control_interval = A2CHelper.num_control_intervals(p)
        sampled_data = A2CHelper.sample_veh_positions(p)
        p.env.loaded_veh_data = sampled_data
        p.env.new_random_game()

        for interval in range(1, num_control_interval + 1):
            if interval > 1:
                p.env.renew_positions_by_file(interval)
                p.env.renew_channel()
                p.env.renew_queue()

            for t in range(p.n_step_per_episode_communication):
                if p.if_fastFading:
                    p.env.renew_fast_fading()

                actions = []
                RRA_all_agents = np.zeros([p.n_veh - 1, p.n_neighbor, 2], dtype="int32")
                environ_helper = EnvironHelper(p)

                global_state = p.env.get_state([0, 0], 0, t)
                global_state = th.tensor(global_state, dtype=th.float32, device=device).squeeze()

                if p.env_name == "POSIG":
                    observations = []
                else:
                    observations = None

                for a in range(p.num_agents):
                    if not p.no_sharing:
                        action = self._select_action_ps(
                            a=a,
                            global_state=global_state,
                            observations=observations,
                            hidden_state=hidden_state,
                            prev_actions=prev_actions,
                            t=t,
                        )
                    else:
                        action = self._select_action_ns(a, global_state)

                    actions.append(action.item())
                    RRA_all_agents[a, 0, 0], RRA_all_agents[a, 0, 1] = \
                        environ_helper.mapping_action2RRA(action)

                joint_action = actions
                global_reward, _, V2I_throughput, done = p.env.step(RRA_all_agents.copy(), t, interval)

                if p.game_mode == 1:
                    global_reward = global_reward[0, 0] + sum(V2I_throughput)
                else:
                    global_reward = global_reward[0, 0]

                if p.env_name == "POSIG":
                    buffer.append((global_state, observations, joint_action, global_reward))
                else:
                    buffer.append((global_state, joint_action, global_reward))

                total_rewards += global_reward

        rtrns = A2CHelper.compute_returns_from_buffer(buffer, done, p.gamma)
        return buffer, rtrns, total_rewards

    def _select_action_ps(self, a, global_state, observations, hidden_state, prev_actions, t):
        p = self.params
        actor_shared = self.actor_shared

        agent_id = F.one_hot(th.tensor(a, device=device), num_classes=p.num_agents).float()

        if p.env_name == "POSIG":
            observation = p.env.get_observation([a, 0], 0, t)
            observation = th.tensor(observation, dtype=th.float32, device=device).squeeze()
            observations.append(observation)

            if p.rnn:
                if p.prev_action_input:
                    prev_a = prev_actions[a].squeeze(0)
                    actor_input = th.cat([observation, agent_id, prev_a], dim=-1)
                else:
                    actor_input = th.cat([observation, agent_id], dim=-1)

                logits, h_a = actor_shared(actor_input, hidden_state[a])
                hidden_state[a] = h_a
            else:
                actor_input = th.cat([observation, agent_id], dim=-1)
                logits = actor_shared(actor_input)
        else:
            actor_input = th.cat([global_state, agent_id], dim=-1)
            logits = actor_shared(actor_input)

        if p.action_masking:
            queue_a = p.env.queue.flatten()[a]
            action, _ = actor_shared.action_sampler(logits, queue_a)
        else:
            action, _ = actor_shared.action_sampler(logits)

        if p.env_name == "POSIG" and p.rnn and p.prev_action_input:
            prev_actions[a] = F.one_hot(action, num_classes=p.action_dim).float().unsqueeze(0).to(device)

        return action

    def _select_action_ns(self, a, global_state):
        p = self.params
        logits = self.actors[a](global_state)
        if p.action_masking:
            queue_a = p.env.queue.flatten()[a]
            action, _ = self.actors[a].action_sampler(logits, queue_a)
        else:
            action, _ = self.actors[a].action_sampler(logits)
        return action

    # ----------------------------------------------------- #
    #  Testing
    # ----------------------------------------------------- #
    def _run_test(self):
        p = self.params
        if (self.episode - 1) % p.test_interval != 0:
            return

        if p.no_sharing:
            actor_states = [a.state_dict() for a in self.actors]
            test_reward = MAA2Ctester.test_MAA2C(self.actors, p)
            for a, s in zip(self.actors, actor_states):
                a.load_state_dict(s)
        else:
            actor_state = self.actor_shared.state_dict()
            test_reward = MAA2Ctester.test_MAA2C(self.actor_shared, p)
            self.actor_shared.load_state_dict(actor_state)

        A2CHelper.finalize_test(self, test_reward, algo_name="MAA2C")

    # ----------------------------------------------------- #
    #  Update routing
    # ----------------------------------------------------- #
    def _update_from_batch(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        p = self.params
        if (not p.no_sharing) and p.env_name == "POSIG" and p.rnn:
            self._update_rnn_case(batch_global_state, batch_observations, batch_joint_actions, batch_rtrns)
        else:
            self._update_fnn_case(batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs)

    # ----------------------------------------------------- #
    #  RNN case
    # ----------------------------------------------------- #
    def _update_rnn_case(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns):
        p = self.params
        b = p.batch_size
        T = p.n_step_per_episode_communication

        global_seq = batch_global_state.view(b, T, -1)
        act_seq = batch_joint_actions.view(b, T, p.num_agents)
        rtrn_seq = batch_rtrns.view(b, T)
        obs_seq = batch_observations.view(b, T, p.num_agents, -1)

        if p.prev_action_input:
            prev_joint_seq = th.zeros(b, T, p.num_agents, p.action_dim, device=device, dtype=th.float32)
            if T > 1:
                prev_idx = act_seq[:, :-1, :].long()
                prev_onehot = F.one_hot(prev_idx, num_classes=p.action_dim).float()
                prev_joint_seq[:, 1:, :, :] = prev_onehot
            prev_joint_flat = prev_joint_seq.view(b, T, p.num_agents * p.action_dim)
            critic_input_seq = th.cat([global_seq, prev_joint_flat], dim=-1)
        else:
            critic_input_seq = global_seq

        # critic
        self.opt_critic.zero_grad()
        h0_critic = th.zeros(1, b, p.critic_hidden_dim, device=device)
        V_seq, _ = self.critic(critic_input_seq, h0_critic)
        V_seq = V_seq.squeeze(-1)
        critic_loss = (rtrn_seq - V_seq).pow(2).mean()
        critic_loss.backward()
        self.opt_critic.step()

        with th.no_grad():
            V_det, _ = self.critic(critic_input_seq, h0_critic)
            V_det = V_det.squeeze(-1)

        advantages = rtrn_seq - V_det
        if p.adv_normalization:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        # actor
        self.opt_actor_shared.zero_grad()
        total_actor_loss = 0.0
        h0_actor = th.zeros(1, b, p.actor_hidden_dim, device=device)

        for a in range(p.num_agents):
            agent_base_seq = obs_seq[:, :, a, :]
            agent_idx = th.full((b, T), a, dtype=th.long, device=device)
            agent_id_seq = F.one_hot(agent_idx, num_classes=p.num_agents).float()

            if p.prev_action_input:
                prev_a_seq = prev_joint_seq[:, :, a, :]
                actor_input_seq = th.cat([agent_base_seq, agent_id_seq, prev_a_seq], dim=-1)
            else:
                actor_input_seq = th.cat([agent_base_seq, agent_id_seq], dim=-1)

            logits_seq, _ = self.actor_shared(actor_input_seq, h0_actor)

            if p.action_masking:
                queue_seq = agent_base_seq[:, :, -1]
                done_mask = (queue_seq <= 0).unsqueeze(-1)
                if done_mask.any():
                    logits_seq = logits_seq.clone()
                    logits_seq[..., :-1] = logits_seq[..., :-1].masked_fill(done_mask, -1e8)

            dist = Categorical(F.softmax(logits_seq, dim=-1))
            actions_a = act_seq[:, :, a]
            logp = dist.log_prob(actions_a)
            ent = dist.entropy()

            loss_a = -(logp * advantages).mean() - p.tau * ent.mean()
            total_actor_loss += loss_a

        total_actor_loss = total_actor_loss / p.num_agents
        total_actor_loss.backward()
        self.opt_actor_shared.step()

    # ----------------------------------------------------- #
    #  FNN case
    # ----------------------------------------------------- #
    def _update_fnn_case(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        p = self.params
        B = batch_joint_actions.size(0)

        critic_input = batch_global_state  # always (no prev_joint in FNN)

        # critic
        self.opt_critic.zero_grad()
        V = self.critic(critic_input).squeeze(-1)
        critic_loss = (batch_rtrns - V).pow(2).mean()
        critic_loss.backward()
        self.opt_critic.step()

        if p.env_name != "POSIG":
            queues = batch_global_state[:, -p.num_agents:]

        with th.no_grad():
            V_det = self.critic(critic_input).squeeze(-1)

        advantages = batch_rtrns - V_det
        if p.adv_normalization:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        # actor
        if p.no_sharing:
            for a in range(p.num_agents):
                logits = self.actors[a](batch_global_state.to(device))

                if p.action_masking:
                    done_mask = (queues[:, a] <= 0)
                    if done_mask.any():
                        logits = logits.clone()
                        logits[done_mask, :-1] = -1e8

                dist = Categorical(F.softmax(logits, dim=-1))
                actions = batch_joint_actions[:, a].to(device)

                loss = -(dist.log_prob(actions) * advantages).mean() - p.tau * dist.entropy().mean()

                self.opt_actors[a].zero_grad()
                loss.backward()
                self.opt_actors[a].step()

        else:
            self.opt_actor_shared.zero_grad()
            total_loss = 0.0

            for a in range(p.num_agents):
                agent_id = F.one_hot(th.tensor(a, device=device), num_classes=p.num_agents).float()
                agent_id = agent_id.unsqueeze(0).repeat(B, 1)

                if has_obs and p.env_name == "POSIG":
                    agent_base = batch_observations[:, a, :].to(device)
                    queue = agent_base[:, -1]
                else:
                    agent_base = batch_global_state.to(device)
                    if p.env_name != "POSIG":
                        queue = queues[:, a]

                actor_input = th.cat([agent_base, agent_id], dim=-1)
                logits = self.actor_shared(actor_input)

                if p.action_masking:
                    done_mask = (queue <= 0)
                    if done_mask.any():
                        logits = logits.clone()
                        logits[done_mask, :-1] = -1e8

                dist = Categorical(F.softmax(logits, dim=-1))
                actions = batch_joint_actions[:, a].to(device)

                loss_a = -(dist.log_prob(actions) * advantages).mean() - p.tau * dist.entropy().mean()
                total_loss += loss_a

            total_loss = total_loss / p.num_agents
            total_loss.backward()
            self.opt_actor_shared.step()