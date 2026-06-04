import csv
from datetime import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import platform
import os

from Networks.Agents.idql_agent import DQNAgent
from Helpers.plotting_helper import plot_test_returns
from Helpers.stat_helper import compute_random_baseline, compute_greedy_baseline, compute_exhaustive_baseline
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
    def train_IDQL_ParameterSharing(env, env_name, env_params, test_data_list,
                                     is_hysteretic_q, algo_params):
        raise ValueError("IDQL PS not finished yet")


class IDQLtrainerNS:
    @staticmethod
    def train_IDQL_NoSharing(
        env,
        env_name,
        env_params,
        test_data_list,
        is_hysteretic_q,
        algo_params,
        algo_name="IDQL",
        trial_run=0,
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

        train_data = env_params.train_data

        # --- CSV logging setup ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = build_csv_name(
            algo_name=algo_name,
            task_type=env_params.task_type,
            n_agent=env.n_agent,
            n_sc=env_params.n_sc,
            ff_tag=env_params.fast_fading_tag,
            trial_run=trial_run,
            ts=ts,
            loc=env_params.loc,
        )
        out_dir = os.path.join("Results", algo_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_file = open(os.path.join(out_dir, csv_name), "w", newline="")
        csv_writer = csv.writer(csv_file)


        # env.new_random_game()

        # all_location_returns, avg_random_return = compute_random_baseline(
        #     env, 
        #     test_data_list, 
        #     num_trials_per_location=10
        # )


        # greedy_returns, avg_greedy_return = compute_greedy_baseline(
        #     env, 
        #     test_data_list, 
        #     num_trials_per_location=1,
        #     top_k_per_agent=5,
        #     local_search_passes=1,
        # )


        # exhaustive_returns, avg_exhaustive_return = compute_exhaustive_baseline(
        #     env, 
        #     test_data_list, 
        #     num_trials_per_location=1,
        #     print_per_step=False,  # Set True for detailed output
        # )



        # === Main Training Loop ===
        for te in range(algo_params.training_episodes):
            total_rewards = 0

            # --- Compute epsilon for this episode ---
            if te < epsi_anneal_episodes:
                epsi = epsi_start - te * (epsi_start - epsi_final) / (epsi_anneal_episodes - 1)
            else:
                epsi = epsi_final

            # --- Sample data based on task type ---
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

            # --- test episodes ---
            if te % algo_params.test_interval == 0:
                test_reward, joint_action = IDQLtester.test_IDQL_NoSharing(
                    agent_list, env_params, algo_params.num_test_episodes, env.n_agent, test_data_list, te
                )
                if prev_joint_action == []:
                    prev_joint_action = joint_action

                test_rewards.append(test_reward)
                csv_writer.writerow([test_reward])
                csv_file.flush()

                plot_test_returns(
                    test_rewards, title="Test Return Over Time IDQL", figure_id=1, pause=1.0,
                )

                print(f"Training reward at episode {te + 1}: {test_reward:.2f}")

            for t in range(env_params.n_step_per_episode):
                # _renew_fast_fading() is called inside env.step(); no explicit call needed.

                # --- Get states ---
                # POSIG: each agent has a distinct local observation.
                # NFIG/SIG: get_state() ignores ag_idx — call once and share.
                if env_name == "POSIG":
                    ag_state_list = [env.get_state(ag_idx, t) for ag_idx in range(len(agent_list))]
                else:
                    shared_state = env.get_state(0, t)
                    ag_state_list = [shared_state] * len(agent_list)

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

                pre_empty_mask = (env.queue <= 0).squeeze(axis=1)
                actions = np.array([a.item() for a in ag_action_list])
                tx_mask = actions != (env.n_actions - 1)

                global_reward, done = env.step(RRA_all_agents.copy(), t)

                _post_empty_tx_mask = pre_empty_mask & tx_mask

                total_rewards += global_reward

                # --- Get next states ---
                if env_name == "POSIG":
                    ag_next_state_list = [env.get_state(ag_idx, t + 1) for ag_idx in range(len(agent_list))]
                else:
                    shared_next_state = env.get_state(0, t + 1)
                    ag_next_state_list = [shared_next_state] * len(agent_list)

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

        csv_file.close()
        return episode_rewards, test_rewards