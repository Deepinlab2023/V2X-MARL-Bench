import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.a2c_actor import A2CSharedActor, A2CActorNS
from Networks.Critics.a2c_critic import A2CSharedCritic, A2CCriticNS
from Helpers.a2c_helper import A2CHelper
from Benchmarkers.ia2c_test import IA2Ctester

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class IA2CTrainer:
    @staticmethod
    def train_IA2C(params):
        trainer = IA2CTrainer(params)
        return trainer.train()

    def __init__(self, params):
        self.params = params
        self._compatibility_checks()
        self.csv_file, self.csv_writer = self._init_logging()
        (
            self.actor_shared,
            self.actors,
            self.critic_shared,
            self.critics,
            self.opt_actor_shared,
            self.opt_actors,
            self.opt_critic_shared,
            self.opt_critics,
        ) = self._init_networks_and_optimizers()

        self.episode_rewards = []
        self.test_rewards = []
        self.episode = 0

    def train(self):
        return A2CHelper.train_loop(self, ctde=False)

    def _compatibility_checks(self):
        A2CHelper.basic_a2c_compat_checks(self.params, algo_name="IA2C")

    def _init_logging(self):
        return A2CHelper.init_csv_logging(self.params, algo_name="IA2C", posig_tag=None)

    def _init_networks_and_optimizers(self):
        p = self.params

        if p.no_sharing:
            actors = [A2CActorNS(p.state_dim, p.action_dim, p.actor_hidden_dim).to(device)
                      for _ in range(p.n_agent)]
            critics = [A2CCriticNS(p.state_dim, p.critic_hidden_dim, p.value_dim).to(device)
                       for _ in range(p.n_agent)]
            actor_shared = None
            critic_shared = None
        else:
            if p.task_type == "POSIG":
                base_actor_dim = p.observation_dim
                base_critic_dim = p.observation_dim
            else:
                base_actor_dim = p.state_dim
                base_critic_dim = p.state_dim

            # Shared actor input: base + agent_id
            actor_input_dim = base_actor_dim + p.n_agent

            actor_shared = A2CSharedActor(actor_input_dim, p).to(device)
            critic_shared = A2CSharedCritic(base_critic_dim, p).to(device)

            actors = None
            critics = None

        if p.no_sharing:
            opt_actors = [th.optim.Adam(a.parameters(), lr=p.alpha) for a in actors]
            opt_critics = [th.optim.Adam(c.parameters(), lr=p.beta) for c in critics]
            opt_actor_shared = None
            opt_critic_shared = None
        else:
            opt_actor_shared = th.optim.Adam(actor_shared.parameters(), lr=p.alpha)
            opt_critic_shared = th.optim.Adam(critic_shared.parameters(), lr=p.beta)
            opt_actors = None
            opt_critics = None

        return (
            actor_shared,
            actors,
            critic_shared,
            critics,
            opt_actor_shared,
            opt_actors,
            opt_critic_shared,
            opt_critics,
        )

    # ----------------------------------------------------- #
    #  Episode rollout + returns
    # ----------------------------------------------------- #
    def _run_single_episode(self):
        p = self.params
        total_rewards = 0.0
        buffer = []
        done = False

        num_control_interval = A2CHelper.num_control_intervals(p)
        sampled_data = A2CHelper.sample_veh_positions(p)

        p.env.train_data = sampled_data
        p.env.new_random_game()

        for interval in range(1, num_control_interval + 1):
            if interval > 1:
                p.env._update_positions_from_data(interval)
                p.env._renew_channels()
                p.env.renew_queue()

            for t in range(p.n_step_per_episode):
                if p.fast_fading_enabled:
                    p.env._renew_fast_fading()

                actions = []
                RRA_all_agents = np.zeros([p.n_agent, 1, 2], dtype="int32")

                if p.task_type == "POSIG":
                    observations = []
                    global_state = None
                else:
                    global_state = p.env.get_state(0, t)
                    global_state = th.tensor(global_state, dtype=th.float32, device=device).squeeze()
                    observations = None

                for a in range(p.n_agent):
                    if not p.no_sharing:
                        action = self._select_action_ps(
                            a=a,
                            global_state=global_state,
                            observations=observations,
                            t=t,
                        )
                    else:
                        action = self._select_action_ns(a, global_state)

                    actions.append(action.item())
                    sc_idx, power_idx = p.env.map_action_to_rra(action, agent_idx=a)
                    RRA_all_agents[a, 0, 0] = sc_idx
                    RRA_all_agents[a, 0, 1] = power_idx

                joint_action = actions
                global_reward, done = p.env.step(RRA_all_agents.copy(), t)

                # Extract scalar reward
                global_reward = global_reward[0, 0]

                if p.task_type == "POSIG":
                    buffer.append((observations, joint_action, global_reward))
                else:
                    buffer.append((global_state, joint_action, global_reward))

                total_rewards += global_reward

        rtrns = A2CHelper.compute_returns_from_buffer(buffer, done, p.gamma)
        return buffer, rtrns, total_rewards

    def _select_action_ps(self, a, global_state, observations, t):
        p = self.params
        actor_shared = self.actor_shared

        agent_id = F.one_hot(th.tensor(a, device=device), num_classes=p.n_agent).float()

        if p.task_type == "POSIG":
            observation = p.env.get_state(a, t)
            observation = th.tensor(observation, dtype=th.float32, device=device).squeeze()
            observations.append(observation)

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

        return action

    def _select_action_ns(self, a, global_state):
        p = self.params
        actor = self.actors[a]
        logits = actor(global_state)

        if p.action_masking:
            queue_a = p.env.queue.flatten()[a]
            action, _ = actor.action_sampler(logits, queue_a)
        else:
            action, _ = actor.action_sampler(logits)

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
            test_reward = IA2Ctester.test_IA2C(self.actors, p)
            for a, s in zip(self.actors, actor_states):
                a.load_state_dict(s)
        else:
            actor_state = self.actor_shared.state_dict()
            test_reward = IA2Ctester.test_IA2C(self.actor_shared, p)
            self.actor_shared.load_state_dict(actor_state)

        A2CHelper.finalize_test(self, test_reward, algo_name="IA2C")

    # ----------------------------------------------------- #
    #  Update routing
    # ----------------------------------------------------- #
    def _update_from_batch(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        self._update_fnn_case(batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs)

    # ----------------------------------------------------- #
    #  FNN case
    # ----------------------------------------------------- #
    def _update_fnn_case(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        p = self.params
        B = batch_joint_actions.size(0)

        if p.task_type != "POSIG":
            queues = batch_global_state[:, -p.n_agent:]

        # ----- critic update -----
        if p.no_sharing:
            for a in range(p.n_agent):
                self.opt_critics[a].zero_grad()
                V_a = self.critics[a](batch_global_state).squeeze(-1)
                loss = (batch_rtrns - V_a).pow(2).mean()
                loss.backward()
                self.opt_critics[a].step()
        else:
            self.opt_critic_shared.zero_grad()
            total_loss = 0.0

            for a in range(p.n_agent):
                if has_obs and p.task_type == "POSIG":
                    agent_base = batch_observations[:, a, :].to(device)
                else:
                    agent_base = batch_global_state.to(device)

                agent_id = F.one_hot(th.tensor(a, device=device), num_classes=p.n_agent).float()
                agent_id = agent_id.unsqueeze(0).repeat(B, 1)

                critic_input = th.cat([agent_base, agent_id], dim=-1)
                V_a = self.critic_shared(critic_input).squeeze(-1)

                total_loss += (batch_rtrns - V_a).pow(2).mean()

            total_loss = total_loss / p.n_agent
            total_loss.backward()
            self.opt_critic_shared.step()

        # ----- actor update -----
        if p.no_sharing:
            for a in range(p.n_agent):
                actor = self.actors[a]
                critic = self.critics[a]

                with th.no_grad():
                    V_agent = critic(batch_global_state).squeeze(-1)
                advantages = (batch_rtrns - V_agent).detach()

                if p.adv_normalization:
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

                logits = actor(batch_global_state.to(device))

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
            total_actor_loss = 0.0

            for a in range(p.n_agent):
                if has_obs and p.task_type == "POSIG":
                    agent_base = batch_observations[:, a, :].to(device)
                    queue = agent_base[:, -1]
                else:
                    agent_base = batch_global_state.to(device)
                    if p.task_type != "POSIG":
                        queue = queues[:, a]

                agent_id = F.one_hot(th.tensor(a, device=device), num_classes=p.n_agent).float()
                agent_id = agent_id.unsqueeze(0).repeat(B, 1)

                actor_input = th.cat([agent_base, agent_id], dim=-1)

                with th.no_grad():
                    V_agent = self.critic_shared(actor_input).squeeze(-1)
                advantages = (batch_rtrns - V_agent).detach()

                if p.adv_normalization:
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

                logits = self.actor_shared(actor_input)

                if p.action_masking:
                    done_mask = (queue <= 0)
                    if done_mask.any():
                        logits = logits.clone()
                        logits[done_mask, :-1] = -1e8

                dist = Categorical(F.softmax(logits, dim=-1))
                actions = batch_joint_actions[:, a].to(device)

                loss_a = -(dist.log_prob(actions) * advantages).mean() - p.tau * dist.entropy().mean()
                total_actor_loss += loss_a

            total_actor_loss = total_actor_loss / p.n_agent
            total_actor_loss.backward()
            self.opt_actor_shared.step()