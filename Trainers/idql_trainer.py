import numpy as np
import torch as th
import matplotlib.pyplot as plt
import platform
import os

from Networks.Agents.idql_agent import DQNAgent
from Benchmarkers.idql_test import *
from Environment.environment_utility import *

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# Linux deterministic fixes
if platform.system() == "Linux":
    print("Applying Linux deterministic fixes...")
    th.set_num_threads(1)
    th.use_deterministic_algorithms(True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


class IDQLtrainerPS:
    @staticmethod
    def train_IDQL_ParameterSharing(trial_run, env, env_name, env_params, test_data_list,
                                     is_hysteretic_q, algo_params):
        raise ValueError("IDQL PS not finished yet")


class IDQLtrainerNS:
    @staticmethod
    def train_IDQL_NoSharing(
        trial_run,
        env,
        env_name,
        env_params,
        test_data_list,
        is_hysteretic_q,
        algo_params,
    ):
        # --- Determine input dimension based on task type ---
        if env_name == "POSIG":
            input_dim = env.local_state_dim
        else:
            input_dim = env.state_dim

        # --- Agents ---
        agent_list = []
        for veh_idx in range(env.n_agent):
            agent_list.append(
                DQNAgent(
                    ag_idx=veh_idx,
                    num_agents=env.n_agent,
                    state_dim=input_dim,
                    action_dim=env.n_actions,
                    is_hysteretic_q=is_hysteretic_q,
                    memory_capacity=algo_params.memory_capacity,
                    batch_size=algo_params.batch_size,
                    gamma=algo_params.gamma,
                    tau=algo_params.tau,
                    lr=algo_params.lr,
                    hidden_dim=algo_params.hidden_dim,
                    hysteretic_high_lr=algo_params.hysteretic_high_lr,
                    hysteretic_low_lr=algo_params.hysteretic_low_lr,
                    force_nt_when_empty=algo_params.force_nt_when_empty,
                )
            )

        # --- Epsilon schedule (linear decay over 80% of training) ---
        epsi_start = 1.0
        epsi_final = 0.05
        epsi_anneal_episodes = int(0.8 * algo_params.training_episodes)

        episode_rewards = []
        test_rewards = []

        prev_joint_action = []
        joint_action_over_te = []

        max_joint_action_reward = float("-inf")
        min_joint_action_reward = float("inf")
        explored_joint_actions = set()

        train_data = env_params.train_data

        print("n_step_per_episode:", env_params.n_step_per_episode)

        for te in range(algo_params.training_episodes):
            total_rewards = 0

            # --- Compute epsilon for this episode ---
            if te < epsi_anneal_episodes:
                epsi = epsi_start - te * (epsi_start - epsi_final) / (epsi_anneal_episodes - 1)
            else:
                epsi = epsi_final

            # Sample data based on task type
            if env_name == "NFIG" and env_params.loc is not None:
                sampled_data = sample_veh_position_from_timestep(train_data, env_params.loc)
            elif env_name in ["SIG", "POSIG"] and env_params.loc is None:
                sampled_data = random_sample(1, train_data)
            elif env_name in ["SIG", "POSIG"] and env_params.loc is not None:
                sampled_data = sample_veh_position_from_timestep(train_data, env_params.loc)
            else:
                raise ValueError(f"Invalid env setup: env_name={env_name}, loc={env_params.loc}")

            env.train_data = sampled_data
            env.new_random_game()

            if te % algo_params.test_interval == 0:
                test_reward, joint_action = IDQLtester.test_IDQL_NoSharing(
                    agent_list, env_params, algo_params.num_test_episodes, env.n_agent, test_data_list, te
                )
                if prev_joint_action == []:
                    prev_joint_action = joint_action

                test_rewards.append(test_reward)

                plt.figure(1)
                plt.clf()
                plt.plot(test_rewards, label="Test Reward")
                plt.xlabel("Test Interval")
                plt.ylabel("Return")
                plt.title("Test Return Over Time IDQL")
                plt.legend()
                plt.grid(True)
                plt.draw()
                plt.pause(1)

                print(f"Training reward at episode {te + 1}: {test_reward:.2f}")

            for t in range(env_params.n_step_per_episode):
                if env_params.fast_fading_enabled:
                    env._renew_fast_fading()

                # --- Get states (POSIG uses per-agent observation) ---
                ag_state_list = []
                for ag_idx in range(len(agent_list)):
                    ag_state = env.get_state(ag_idx, t)
                    ag_state_list.append(ag_state)

                ag_action_list = []
                joint_action = []

                # RRA_all_agents shape: [n_agents, 1, 2]
                #   - Axis 0: agent index
                #   - Axis 1: V2V link index (always 1 link per agent in current design)
                #   - Axis 2: [subchannel_index, power_level_index]
                RRA_all_agents = np.zeros([len(agent_list), 1, 2], dtype="int32")

                greedy_joint_action = []
                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].eps_threshold = 0
                    action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                    greedy_joint_action.append(action.item())

                    # Map discrete action to (subchannel, power_level) tuple
                    # RRA_all_agents[ag_idx, 0, 0] = subchannel index (or -1 for no transmission)
                    # RRA_all_agents[ag_idx, 0, 1] = power level index (or -1 for no transmission)
                    RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                        action, agent_idx=ag_idx
                    )
                joint_action_over_te.append(greedy_joint_action)

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].eps_threshold = epsi
                    action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                    ag_action_list.append(np.array([[action.item()]]))
                    joint_action.append(action)
                    RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                        action, agent_idx=ag_idx
                    )

                explored_joint_actions.add(tuple(a.item() for a in joint_action))

                pre_empty_mask = (env.queue <= 0).squeeze(axis=1)
                actions = np.array([a.item() for a in ag_action_list])
                tx_mask = actions != (env.n_actions - 1)

                global_reward, done = env.step(RRA_all_agents.copy(), t)

                _post_empty_tx_mask = pre_empty_mask & tx_mask

                total_rewards += global_reward

                if global_reward > max_joint_action_reward:
                    max_joint_action_reward = global_reward
                if global_reward < min_joint_action_reward:
                    min_joint_action_reward = global_reward

                # --- Get next states (POSIG uses per-agent observation) ---
                ag_next_state_list = []
                for ag_idx in range(len(agent_list)):
                    ag_next_state = env.get_state(ag_idx, t + 1)
                    ag_next_state_list.append(ag_next_state)

                for ag_idx in range(len(agent_list)):
                    transition = (
                        ag_state_list[ag_idx],
                        ag_action_list[ag_idx],
                        ag_next_state_list[ag_idx],
                        done,
                        global_reward,
                    )
                    agent_list[ag_idx].store_transition(*transition)

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].optimize_model()

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].soft_update_target_net()

            episode_rewards.append(np.mean(total_rewards))

        print(f"\nMax joint action reward seen during training: {float(max_joint_action_reward):.2f}")
        print(f"Min joint action reward seen during training: {float(min_joint_action_reward):.2f}")
        print(f"Unique joint actions explored during training: {len(explored_joint_actions)}")

        return episode_rewards, test_rewards