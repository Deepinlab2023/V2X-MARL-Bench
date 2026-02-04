import numpy as np
import torch as th

from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class MAA2Ctester:
    @staticmethod
    def test_MAA2C(policy, params):
        """
        policy:
          - if params.no_sharing == False: shared actor (A2CSharedActor)
          - if params.no_sharing == True:  list of A2CActorNS, one per agent
        """
        tester = MAA2Ctester(policy, params)
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
        p = self.params

        # Mirror trainer constraint: NS not supported for POSIG
        if p.no_sharing and p.task_type == "POSIG":
            raise ValueError("MAA2C No-Sharing is not supported for POSIG testing.")

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

        # Load test veh data
        test_data = p.test_data_list[idx % len(p.test_data_list)]
        env.train_data = test_data
        env.new_random_game()

        for t in range(p.n_step_per_episode):
            if p.fast_fading_enabled:
                env._renew_fast_fading()

            RRA_all_agents = np.zeros([p.n_agent, 1, 2], dtype="int32")

            # For FO (NFIG/SIG), get_state returns global state
            # For POSIG, get_state returns local obs, so use get_global_state for consistency
            if p.task_type == "POSIG":
                global_state = env.get_global_state(t)
            else:
                global_state = env.get_state(0, t)
            global_state = th.tensor(global_state, dtype=th.float32, device=device).squeeze()

            # Select actions
            for a in range(p.n_agent):
                if p.no_sharing:
                    action = self._select_action_ns(a, global_state)
                else:
                    action = self._select_action_ps(a, global_state, t)

                sc_idx, power_idx = env.map_action_to_rra(action, agent_idx=a)
                RRA_all_agents[a, 0, 0] = sc_idx
                RRA_all_agents[a, 0, 1] = power_idx

            # Environment step
            global_reward, done = env.step(RRA_all_agents.copy(), t)
            total_rewards += global_reward[0, 0]

        return total_rewards

    # ==========================
    #   ACTION SELECTION (PS)
    # ==========================
    def _select_action_ps(self, a, global_state, t):
        """
        Parameter sharing case:
        - FO (NFIG/SIG): use global_state + agent_id
        - POSIG: use observation + agent_id
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
        actor = self.policy[a]

        logits = actor(global_state)

        if p.action_masking:
            queue_a = env.queue.flatten()[a]
            action, _ = actor.action_sampler(logits, queue_a)
        else:
            action, _ = actor.action_sampler(logits)

        return action