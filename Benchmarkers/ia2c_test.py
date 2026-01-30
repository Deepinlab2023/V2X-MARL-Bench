import numpy as np
import torch as th

from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class IA2Ctester:
    @staticmethod
    def test_IA2C(policy, params):
        """
        policy:
          - if params.no_sharing == False: shared actor (SharedA2CActor)
          - if params.no_sharing == True:  list/sequence of A2CActorNS, one per agent
        """
        tester = IA2Ctester(policy, params)
        return tester.test()

    # ==========================
    #   INIT / SETUP
    # ==========================
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params
        self._sanity_checks()
        self.env = Environ(params.env_params)

    def _sanity_checks(self):
        # mirror trainer constraint
        if self.params.no_sharing and self.params.task_type == "POSIG":
            raise ValueError("IA2C No-Sharing is not supported for POSIG testing.")

    # ==========================
    #   MAIN TEST LOOP
    # ==========================
    def test(self):
        p = self.params
        test_rewards = np.zeros(p.num_test_episodes)

        for i in range(p.num_test_episodes):
            total_reward = self._run_single_episode(i)
            test_rewards[i] = total_reward

        return float(np.mean(test_rewards))

    # ==========================
    #   SINGLE EPISODE ROLLOUT
    # ==========================
    def _run_single_episode(self, idx):
        p = self.params
        env = self.env

        total_rewards = 0.0

        # POSIG PS-only RNN / prev-action init (same logic as trainer)
        if (not p.no_sharing) and p.task_type == "POSIG":
            hidden_state, prev_actions = self._init_posig_states()
        else:
            hidden_state, prev_actions = None, None

        # Load test veh data
        test_data = p.test_data_list[idx % len(p.test_data_list)]
        env.train_data = test_data
        env.new_random_game()

        # Refactor note:
        # t_max_control is obsolete. Testing is one continuous rollout over n_step_per_episode.
        for t in range(p.n_step_per_episode):
            if p.fast_fading_enabled:
                env._renew_fast_fading()

            RRA_all_agents = np.zeros([p.n_agent, 1, 2], dtype="int32")

            # FO uses global state; POSIG still uses per-agent observation inside _select_action_ps
            global_state = env.get_state(0, t)
            global_state = th.tensor(global_state, dtype=th.float32, device=device).squeeze()

            # select actions
            for a in range(p.n_agent):
                if p.no_sharing:
                    action = self._select_action_ns(a, global_state)
                else:
                    action, hidden_state, prev_actions = self._select_action_ps(
                        a,
                        global_state,
                        hidden_state,
                        prev_actions,
                        t,
                    )

                sc_idx, power_idx = env.map_action_to_rra(action, agent_idx=a)
                RRA_all_agents[a, 0, 0] = sc_idx
                RRA_all_agents[a, 0, 1] = power_idx

            # env step
            global_reward, done = env.step(RRA_all_agents.copy(), t)
            total_rewards += global_reward[0, 0]

        return total_rewards

    # ==========================
    #   POSIG STATE INIT
    # ==========================
    def _init_posig_states(self):
        p = self.params

        if not ((not p.no_sharing) and p.task_type == "POSIG"):
            return None, None

        if p.rnn:
            hidden_state = [
                th.zeros(1, 1, p.actor_hidden_dim, device=device)
                for _ in range(p.n_agent)
            ]
        else:
            hidden_state = None

        # prev_action only valid in RNN mode
        if p.rnn and p.prev_action_input:
            prev_actions = [
                th.zeros(1, p.action_dim, device=device)
                for _ in range(p.n_agent)
            ]
        else:
            prev_actions = None

        return hidden_state, prev_actions

    # ==========================
    #   ACTION SELECTION (PS)
    # ==========================
    def _select_action_ps(self, a, global_state, hidden_state, prev_actions, t):
        """
        Parameter sharing case:
        - FO (NFIG/SIG): use global_state + agent_id
        - POSIG: use observation + agent_id [+ per-agent prev_action]
        """
        p = self.params
        env = self.env
        actor_shared = self.policy

        agent_id = th.nn.functional.one_hot(
            th.tensor(a, device=device),
            num_classes=p.n_agent,
        ).float()

        if p.task_type == "POSIG":
            observation = env.get_state(a, t)
            observation = th.tensor(observation, dtype=th.float32, device=device).squeeze()

            use_prev = (p.rnn and p.prev_action_input)

            if use_prev:
                prev_action_a = prev_actions[a].squeeze(0)  # [action_dim]
                actor_input = th.cat([observation, agent_id, prev_action_a], dim=-1)
            else:
                actor_input = th.cat([observation, agent_id], dim=-1)

            if p.rnn:
                logits, h_a = actor_shared(actor_input, hidden_state[a])
                hidden_state[a] = h_a
            else:
                logits = actor_shared(actor_input)
        else:
            # FO PS (NFIG / SIG)
            actor_input = th.cat([global_state, agent_id], dim=-1)
            logits = actor_shared(actor_input)

        # masking
        if p.action_masking:
            queue_a = env.queue.flatten()[a]
            action, _ = actor_shared.action_sampler(logits, queue_a)
        else:
            action, _ = actor_shared.action_sampler(logits)

        # update prev_actions if used
        if p.task_type == "POSIG" and p.rnn and p.prev_action_input:
            one_hot_prev = th.nn.functional.one_hot(
                action, num_classes=p.action_dim
            ).float().unsqueeze(0).to(device)
            prev_actions[a] = one_hot_prev

        return action, hidden_state, prev_actions

    # ==========================
    #   ACTION SELECTION (NS)
    # ==========================
    def _select_action_ns(self, a, global_state):
        """
        No sharing: FO only, one actor per agent.
        """
        p = self.params
        env = self.env
        actor = self.policy[a]

        logits = actor(global_state)

        if p.action_masking:
            queue_a = env.queue.flatten()[a]
            action, _ = actor.action_sampler(logits, queue_a)
        else:
            action, _ = actor.action_sampler(logits)

        return action
