import csv
from datetime import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import platform
import os

from Networks.Agents.qmix_agent import QMIXAgent
from Helpers.qmix_helper import QMIX_network_init, QMIXLearner
from Helpers.plotting_helper import plot_test_returns
from Benchmarkers.qmix_test import QMIXtester
from Environment.environment_utility import *

device = th.device("cuda" if th.cuda.is_available() else "cpu")

if platform.system() == "Linux":
    print("Applying Linux deterministic fixes...")
    th.set_num_threads(1)
    th.use_deterministic_algorithms(True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


class QMIXtrainerPS:
    @staticmethod
    def train_QMIX_ParameterSharing(trial_run, env, env_name, env_params, test_data_list,
                                     is_vdn, algo_params):
        raise ValueError("QMIX PS not finished yet")


class QMIXtrainerNS:
    @staticmethod
    def train_QMIX_NoSharing(
        env,
        env_name,
        env_params,
        test_data_list,
        is_vdn,
        algo_params,
        algo_name="QMIX",
        trial_run=0,
    ):
        # --- Determine input dimension based on task type ---
        if env_name == "POSIG":
            input_dim = env.local_state_dim
        else:
            input_dim = env.state_dim

        # === QMIX Agent Initialization ===
        agent_list = []
        for veh_idx in range(env.n_agent):
            agent_list.append(
                QMIXAgent(
                    ag_idx=veh_idx,
                    num_agents=env.n_agent,
                    state_dim=input_dim,
                    action_dim=env.n_actions,
                    gamma=algo_params.gamma,
                    tau=algo_params.tau,
                    hidden_dim=algo_params.hidden_dim,
                    force_nt_when_empty=algo_params.force_nt_when_empty,
                )
            )

        # === QMIX Learner Initialization ===
        mix_args = QMIX_network_init(env, algo_params, env.global_state_dim)

        qmix_learner = QMIXLearner(
            agent_list,
            device,
            is_vdn,
            mix_args,
            memory_capacity=algo_params.memory_capacity,
            batch_size=algo_params.batch_size,
            agent_lr=algo_params.agent_lr,
            mixer_lr=algo_params.mixer_lr,
        )
        qmix_learner.init_model()

        # === Epsilon schedule (linear decay over 80% of training) ===
        epsi_start = 1.0
        epsi_final = 0.05
        epsi_anneal_episodes = int(0.8 * algo_params.training_episodes)

        episode_rewards = []
        test_rewards = []

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
                test_reward = QMIXtester.test_QMIX_NoSharing(
                    agent_list, env_params, algo_params.num_test_episodes, env.n_agent, test_data_list, te
                )
                test_rewards.append(test_reward)
                csv_writer.writerow([test_reward])
                csv_file.flush()

                plot_test_returns(
                    test_rewards, title="Test Return Over Time QMIX", figure_id=1, pause=1.0,
                )
                print(f'Training reward at episode {te + 1}: {test_reward:.2f}')

            for t in range(env_params.n_step_per_episode):
                if env_params.fast_fading_enabled:
                    env._renew_fast_fading()

                # --- Get states (local for POSIG, global for others) ---
                ag_state_list = []
                for ag_idx in range(len(agent_list)):
                    ag_state = env.get_state(ag_idx, t)
                    ag_state_list.append(ag_state)

                # --- Get global state for mixer (always use SIG state) ---
                global_state = env.get_global_state(t)

                # --- Get actions ---
                ag_action_list = []
                RRA_all_agents = np.zeros([len(agent_list), 1, 2], dtype='int32')

                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].eps_threshold = epsi
                    action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                    ag_action_list.append(np.array([[action.item()]]))
                    RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = env.map_action_to_rra(
                        action, agent_idx=ag_idx
                    )

                # --- Step ---
                global_reward, done = env.step(RRA_all_agents.copy(), t)
                total_rewards += global_reward

                # --- Get next states ---
                ag_next_state_list = []
                for ag_idx in range(len(agent_list)):
                    ag_next_state = env.get_state(ag_idx, t + 1)
                    ag_next_state_list.append(ag_next_state)

                # --- Get next global state for mixer ---
                global_next_state = env.get_global_state(t + 1)

                # --- Store transition ---
                qmix_learner.store_transition(
                    ag_state_list, ag_action_list, ag_next_state_list, done, global_reward,
                    global_state, global_next_state
                )

                # --- Train ---
                qmix_learner.centralized_training()

                # --- Soft update ---
                for ag_idx in range(len(agent_list)):
                    agent_list[ag_idx].soft_update_target_net()
                if not is_vdn:
                    qmix_learner.soft_update_target_net()

            episode_rewards.append(np.mean(total_rewards))

        csv_file.close()
        return episode_rewards, test_rewards