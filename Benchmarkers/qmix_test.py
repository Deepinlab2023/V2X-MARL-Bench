import numpy as np
import torch as th
import platform

# Match the refactored IDQL tester: same env module + same API contract
from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class QMIXtester:
    @staticmethod
    def test_QMIX_NoSharing(agent_list, params, num_test_episodes, num_agents, test_data_list, qmix_learner):
        """
        Refactored to match IDQLtester loop:
        - env.train_data = test_data
        - env.get_state(ag_idx, t)
        - env.map_action_to_rra(action, agent_idx)
        - env.step(RRA_all_agents, t) -> (global_reward, done)
        - epsilon forced to 0 for greedy evaluation
        """
        if platform.system() == "Linux":
            th.set_num_threads(1)
            th.use_deterministic_algorithms(True)

        env = Environ(params)
        test_rewards = np.zeros(num_test_episodes)

        for i in range(num_test_episodes):
            total_rewards = 0.0
            test_data = test_data_list[i % len(test_data_list)]

            # Standardized data injection (same as IDQL)
            env.train_data = test_data
            env.new_random_game()

            for t in range(params.n_step_per_episode):
                if getattr(params, "fast_fading_enabled", False):
                    env._renew_fast_fading()

                # --- States ---
                ag_state_list = []
                for ag_idx in range(len(agent_list)):
                    ag_state = env.get_state(ag_idx, t)
                    ag_state_list.append(ag_state)

                # --- Actions -> RRA ---
                RRA_all_agents = np.zeros([len(agent_list), 1, 2], dtype="int32")

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].eps_threshold = 0

                    # Standard interface: select_action(state, env)
                    # (kept try/except in case your QMIXAgent still has legacy signature)
                    try:
                        action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                    except TypeError:
                        action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], params.game_mode, env)

                    # Standard mapping: env owns action->RRA
                    RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                        action, agent_idx=ag_idx
                    )

                # --- Step ---
                # Standard interface: step(RRA, t) -> (global_reward, done)
                # (kept try/except in case your env.step is still legacy in some branches)
                try:
                    global_reward, done = env.step(RRA_all_agents.copy(), t)
                except TypeError:
                    global_reward, _, _, done = env.step(RRA_all_agents.copy(), t, 1)

                total_rewards += float(global_reward[0, 0])

                if done:
                    break

            test_rewards[i] = total_rewards

        return float(np.mean(test_rewards))
