import numpy as np
import torch as th
import platform

from Environment.environment import Environ

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class IDQLtester:
    @staticmethod
    def test_IDQL_NoSharing(agent_list, params, num_test_episodes, num_agents, test_data_list, te):
        if platform.system() == "Linux":
            th.set_num_threads(1)
            th.use_deterministic_algorithms(True)

        env = Environ(params)
        test_rewards = np.zeros(num_test_episodes)

        for i in range(num_test_episodes):
            total_rewards = 0
            test_data = test_data_list[i % len(test_data_list)]

            env.train_data = test_data
            env.new_random_game()

            for t in range(params.n_step_per_episode):
                if params.fast_fading_enabled:
                    env._renew_fast_fading()

                # --- Get states (POSIG uses per-agent observation) ---
                ag_state_list = []
                for ag_idx in range(len(agent_list)):
                    # ag_state = env.get_state([ag_idx, 0], 0, t)
                    ag_state = env.get_state(ag_idx, t)
                    ag_state_list.append(ag_state)

                joint_action = []
                RRA_all_agents = np.zeros([len(agent_list), 1, 2], dtype="int32")

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].eps_threshold = 0
                    action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                    joint_action.append(action)
                    RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                        action, agent_idx=ag_idx
                    )

                # global_reward, _, V2I_throughput, _ = env.step(RRA_all_agents.copy(), t, 1)
                global_reward, done = env.step(RRA_all_agents.copy(), t)


                # # For NFIG, add V2I throughput to reward
                # if params.task_type == "NFIG":
                #     total_rewards += global_reward + sum(V2I_throughput)
                # else:
                #     total_rewards += global_reward[0, 0]
                total_rewards += global_reward[0, 0]

            test_rewards[i] += total_rewards

        average_reward = np.mean(test_rewards)
        return average_reward, joint_action