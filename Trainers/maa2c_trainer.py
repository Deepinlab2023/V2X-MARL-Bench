import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.a2c_actor import A2CSharedActor, A2CActorNS
from Networks.Critics.a2c_critic import A2CCentralizedCritic
from Helpers.a2c_helper import A2CHelper
from Benchmarkers.maa2c_test import MAA2Ctester

from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAA2CTrainer:
    @staticmethod
    def train_MAA2C(params):
        trainer = MAA2CTrainer(params)
        return trainer.train()

    # ==========================
    #   INIT / SETUP
    # ==========================
    def __init__(self, params):
        self.params = params
        self._compatibility_checks()
        self.env = Environ(params.env_params)
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
        return A2CHelper.init_csv_logging(self.params, algo_name="MAA2C", posig_tag=None)

    def _init_networks_and_optimizers(self):
        p = self.params

        if p.no_sharing:
            # NS: one actor per agent, FO only
            actors = [
                A2CActorNS(p.state_dim, p.action_dim, p.actor_hidden_dim).to(device)
                for _ in range(p.n_agent)
            ]
            actor_shared = None
        else:
            # PS: shared actor
            if p.task_type == "POSIG":
                base_actor_dim = p.observation_dim
            else:
                base_actor_dim = p.state_dim

            actor_input_dim = base_actor_dim + p.n_agent

            actor_shared = A2CSharedActor(actor_input_dim, p).to(device)
            actors = None

        # Centralized critic always uses global state
        critic = A2CCentralizedCritic(p).to(device)

        # Optimizers
        if p.no_sharing:
            opt_actors = [th.optim.Adam(a.parameters(), lr=p.alpha) for a in actors]
            opt_actor_shared = None
        else:
            opt_actor_shared = th.optim.Adam(actor_shared.parameters(), lr=p.alpha)
            opt_actors = None

        opt_critic = th.optim.Adam(critic.parameters(), lr=p.beta)

        return actor_shared, actors, critic, opt_actor_shared, opt_actors, opt_critic

    # ==========================
    #   ROLLOUT
    # ==========================
    def _run_single_episode(self):
        p = self.params
        env = self.env

        total_rewards = 0.0
        buffer = []
        done = False

        # Sample and load vehicle positions
        sampled_data = A2CHelper.sample_veh_positions(p)
        env.train_data = sampled_data
        env.new_random_game()

        for t in range(p.n_step_per_episode):
            if p.fast_fading_enabled:
                env._renew_fast_fading()

            actions = []
            RRA_all_agents = np.zeros([p.n_agent, 1, 2], dtype="int32")

            # Global state (used by centralized critic)
            # For POSIG, env.get_state returns local obs, so use get_global_state instead
            if p.task_type == "POSIG":
                global_state = env.get_global_state(t)
            else:
                global_state = env.get_state(0, t)
            global_state = th.tensor(global_state, dtype=th.float32, device=device).squeeze()

            # Collect observations for POSIG
            if p.task_type == "POSIG":
                observations = []
            else:
                observations = None

            # Select actions for all agents
            for a in range(p.n_agent):
                if p.no_sharing:
                    action = self._select_action_ns(a, global_state)
                else:
                    action = self._select_action_ps(
                        a=a,
                        global_state=global_state,
                        observations=observations,
                        t=t,
                    )

                actions.append(action.item())
                sc_idx, power_idx = env.map_action_to_rra(action, agent_idx=a)
                RRA_all_agents[a, 0, 0] = sc_idx
                RRA_all_agents[a, 0, 1] = power_idx

            # Environment step
            joint_action = actions
            global_reward, done = env.step(RRA_all_agents.copy(), t)
            global_reward = global_reward[0, 0]

            # Store transition
            if p.task_type == "POSIG":
                buffer.append((global_state, observations, joint_action, global_reward))
            else:
                buffer.append((global_state, joint_action, global_reward))

            total_rewards += global_reward

        rtrns = A2CHelper.compute_returns_from_buffer(buffer, done, p.gamma)
        return buffer, rtrns, total_rewards

    # ==========================
    #   ACTION SELECTION (PS)
    # ==========================
    def _select_action_ps(self, a, global_state, observations, t):
        """
        Parameter sharing case:
        - FO (NFIG/SIG): use global_state + agent_id
        - POSIG: use observation + agent_id
        """
        p = self.params
        env = self.env
        actor_shared = self.actor_shared

        agent_id = F.one_hot(
            th.tensor(a, device=device),
            num_classes=p.n_agent,
        ).float()

        if p.task_type == "POSIG":
            observation = env.get_state(a, t)
            observation = th.tensor(observation, dtype=th.float32, device=device).squeeze()
            observations.append(observation)

            actor_input = th.cat([observation, agent_id], dim=-1)
        else:
            # FO PS (NFIG / SIG)
            actor_input = th.cat([global_state, agent_id], dim=-1)

        logits = actor_shared(actor_input)

        # Action masking
        if p.action_masking:
            queue_a = env.queue.flatten()[a]
            action, _ = actor_shared.action_sampler(logits, queue_a)
        else:
            action, _ = actor_shared.action_sampler(logits)

        return action

    # ==========================
    #   ACTION SELECTION (NS)
    # ==========================
    def _select_action_ns(self, a, global_state):
        """
        No sharing: FO only, one actor per agent.
        """
        p = self.params
        env = self.env

        logits = self.actors[a](global_state)

        if p.action_masking:
            queue_a = env.queue.flatten()[a]
            action, _ = self.actors[a].action_sampler(logits, queue_a)
        else:
            action, _ = self.actors[a].action_sampler(logits)

        return action

    # ==========================
    #   TESTING
    # ==========================
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

    # ==========================
    #   UPDATE ROUTING
    # ==========================
    def _update_from_batch(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        self._update_fnn_case(batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs)

    # ==========================
    #   FNN UPDATE
    # ==========================
    def _update_fnn_case(self, batch_global_state, batch_observations, batch_joint_actions, batch_rtrns, has_obs):
        p = self.params
        B = batch_joint_actions.size(0)

        # Centralized critic uses global state
        critic_input = batch_global_state

        # Critic update
        self.opt_critic.zero_grad()
        V = self.critic(critic_input).squeeze(-1)
        critic_loss = (batch_rtrns - V).pow(2).mean()
        critic_loss.backward()
        self.opt_critic.step()

        # Extract queues for FO masking
        if p.task_type != "POSIG":
            queues = batch_global_state[:, -p.n_agent:]

        # Compute advantages
        with th.no_grad():
            V_det = self.critic(critic_input).squeeze(-1)

        advantages = batch_rtrns - V_det
        if p.adv_normalization:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        # Actor update
        if p.no_sharing:
            # NS: update each actor separately
            for a in range(p.n_agent):
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
            # PS: shared actor
            self.opt_actor_shared.zero_grad()
            total_loss = 0.0

            for a in range(p.n_agent):
                agent_id = F.one_hot(
                    th.tensor(a, device=device),
                    num_classes=p.n_agent,
                ).float()
                agent_id = agent_id.unsqueeze(0).repeat(B, 1)

                if has_obs and p.task_type == "POSIG":
                    agent_base = batch_observations[:, a, :].to(device)
                    queue = agent_base[:, -1]
                else:
                    agent_base = batch_global_state.to(device)
                    if p.task_type != "POSIG":
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

            total_loss = total_loss / p.n_agent
            total_loss.backward()
            self.opt_actor_shared.step()