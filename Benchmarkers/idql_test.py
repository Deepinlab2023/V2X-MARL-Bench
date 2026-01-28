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

            # All task types use single control interval in current setup
            num_control_interval = 1

            for interval in range(1, num_control_interval + 1):
                if interval > 1:
                    env._update_positions_from_data(interval)
                    env._renew_channels()
                    env.renew_queue()

                epsi = 0


                for t in range(params.n_step_per_episode):
                    if params.fast_fading_enabled:
                        env._renew_fast_fading()

                    ag_state_list = []
                    for ag_idx in range(len(agent_list)):
                        ag_state = env.get_state([0, 0], 0, t)
                        ag_state_list.append(ag_state)


                    joint_action = []
                    RRA_all_agents = np.zeros([len(agent_list), params.n_neighbor, 2], dtype="int32")

                    for ag_idx in range(len(agent_list)):
                        agent_list[ag_idx].eps_threshold = epsi
                        action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                        joint_action.append(action)
                        RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                            action, agent_idx=ag_idx
                        )

                    global_reward, _, V2I_throughput, _ = env.step(RRA_all_agents.copy(), t, interval)

                    # For NFIG, add V2I throughput to reward
                    if params.task_type == "NFIG":
                        total_rewards += global_reward + sum(V2I_throughput)
                    else:
                        total_rewards += global_reward[0, 0]

            test_rewards[i] += total_rewards


        average_reward = np.mean(test_rewards)
        return average_reward, joint_action