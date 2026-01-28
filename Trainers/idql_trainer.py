import numpy as np
import torch as th
import csv
import matplotlib.pyplot as plt
import copy
import platform
import os
import random

from Networks.Agents.idql_agent import DQNAgent
from Benchmarkers.idql_test import *
from Environment.environment_utility import *

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

# --- Checkpoint path config ---
_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_MODELS_DIR = os.path.join(_REPO_ROOT, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

# Linux deterministic fixes
if platform.system() == "Linux":
    print("Applying Linux deterministic fixes...")
    th.set_num_threads(1)
    th.use_deterministic_algorithms(True)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def q_state_dict(agent):
    """Extract Q-network state dict from agent."""
    for attr in ("q_net", "qnetwork", "q_network", "net", "model"):
        mod = getattr(agent, attr, None)
        if mod is not None and hasattr(mod, "state_dict"):
            return mod.state_dict()
    return None


class IDQLtrainerPS:
    @staticmethod
    def train_IDQL_ParameterSharing(trial_run, env, env_name, params, test_data_list,
                                     state_dim, observation_dim, action_dim, num_agents, gamma, actor_hidden_dim,
                                     critic_hidden_dim, value_dim, alpha, beta, tau, test_interval,
                                     num_training_iterations, num_test_episodes, batch_size):
        raise ValueError("IDQL PS not finished yet")


class IDQLtrainerNS:
    @staticmethod
    def train_IDQL_NoSharing(
        trial_run,
        env,
        env_name,
        params,
        test_data_list,
        state_dim,
        observation_dim,
        action_dim,
        num_agents,
        gamma,
        hidden_dim,
        value_dim,
        tau,
        test_interval,
        num_training_iterations,
        num_test_episodes,
        batch_size,
        is_hysteretic_q,
    ):
        # --- CSV log ---
        if env_name == "NFIG":
            csv_file = open(f"IDQLNS_trial_{trial_run}_NFIG_{int(params.loc)}.csv", "a", newline="")
        elif env_name == "SIG":
            if params.loc is None:
                csv_file = open(f"IDQLNS_trial_{trial_run}_SIG_ML_{params.fast_fading_tag}.csv", "a", newline="")
            else:
                csv_file = open(
                    f"IDQLNS_trial_{trial_run}_SIG_{params.fast_fading_tag}_{params.loc}.csv", "a", newline=""
                )
        elif env_name == "POSIG":
            if params.loc is None:
                csv_file = open(f"IDQLNS_trial_{trial_run}_POSIG_ML_{params.fast_fading_tag}.csv", "a", newline="")
            else:
                csv_file = open(
                    f"IDQLNS_trial_{trial_run}_POSIG_{params.fast_fading_tag}_{params.loc}.csv", "a", newline=""
                )
        else:
            raise ValueError(f"Unsupported env_name: {env_name}")

        csv_writer = csv.writer(csv_file)

        # --- Agents ---
        agent_list = []
        for veh_idx in range(num_agents):
            agent_list.append(
                DQNAgent(
                    ag_idx=veh_idx,
                    num_agents=num_agents,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    is_hysteretic_q=is_hysteretic_q,
                )
            )

        # --- Epsilon schedule ---
        epsi_final = 0.05
        epsi_anneal_length = int(0.8 * num_training_iterations)
        epsi_anneal_length_time = epsi_anneal_length * params.t_max

        episode_rewards = []
        test_rewards = []

        prev_joint_action = []
        joint_action_over_te = []

        max_joint_action_reward = float("-inf")
        min_joint_action_reward = float("inf")
        explored_joint_actions = set()

        train_data = params.train_data


        # np.random.seed(7)
        eval_data_list = None
        # if env_name == "SIG" and params.loc is None:
        #     for _ in range(9):
        #         eval_sample = random_sample(1, train_data)
        #         eval_data_list.append(eval_sample)
        # else:
        #     eval_data_list = None

        print("n_step_per_episode:", params.n_step_per_episode)
        print("n_step_per_episode_communication:", getattr(params, "n_step_per_episode_communication", None))
        print("t_max:", params.t_max, "t_max_control:", getattr(params, "t_max_control", None))


        for te in range(num_training_iterations):
            total_rewards = 0
            done = False

            # NFIG and SIG/POSIG all use single control interval in current setup
            num_control_interval = 1

            # Sample data based on task type
            if env_name == "NFIG" and params.loc is not None:
                sampled_data = sample_veh_position_from_timestep(train_data, params.loc)
            elif env_name in ["SIG", "POSIG"] and params.loc is None:
                sampled_data = random_sample(1, train_data)
            elif env_name in ["SIG", "POSIG"] and params.loc is not None:
                sampled_data = sample_veh_position_from_timestep(train_data, params.loc)
            else:
                raise ValueError(f"Invalid env setup: env_name={env_name}, loc={params.loc}")

            env.train_data = sampled_data
            env.new_random_game()

            # print(te)
            # print(sampled_data)

            if te % test_interval == 0:
                test_reward, joint_action = IDQLtester.test_IDQL_NoSharing(
                    agent_list, params, num_test_episodes, num_agents, test_data_list, te
                )
                if prev_joint_action == []:
                    prev_joint_action = joint_action

                test_rewards.append(test_reward)
                csv_writer.writerow([test_reward])
                csv_file.flush()

                # if eval_data_list is not None:
                #     eval_reward, _ = IDQLtester.test_IDQL_NoSharing(
                #         agent_list, params, len(eval_data_list), num_agents, eval_data_list, te
                #     )

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

                if eval_data_list is not None:
                    print(f"Training reward at episode {te + 1}: {test_reward:.2f}, Eval={eval_reward:.2f}")
                else:
                    print(f"Training reward at episode {te + 1}: {test_reward:.2f}")

                ckpt = {
                    "episode": int(te + 1),
                    "env_name": env_name,
                    "trial_run": int(trial_run),
                    "test_reward": float(test_reward),
                    "q_states": [q_state_dict(a) for a in agent_list],
                }
                checkpoint_path = os.path.join(_MODELS_DIR, f"IDQLNS_{env_name}_trial{trial_run}_ep{te+1}.pt")
                th.save(ckpt, checkpoint_path)

            for interval in range(1, num_control_interval + 1):
                if interval > 1:
                    env._update_positions_from_data(interval)
                    env._renew_channels()
                    env.renew_queue()

                t_global = te * params.t_max + interval

                if t_global < epsi_anneal_length_time - 1:
                    epsi = 1 - t_global * (1 - epsi_final) / (epsi_anneal_length_time - 1)
                    epsi_new = 1 - ((te + 1) * params.t_max + interval) * (1 - epsi_final) / (
                        epsi_anneal_length_time - 1
                    )
                else:
                    epsi = epsi_final
                    epsi_new = epsi

                # print("te: ", te, "epsi: ", epsi)

                for t in range(params.n_step_per_episode):
                    if params.fast_fading_enabled:
                        env._renew_fast_fading()

                    ag_state_list = []
                    for _ag_idx in range(len(agent_list)):
                        ag_state = env.get_state([0, 0], 0, t)
                        ag_state_list.append(ag_state)

                    ag_action_list = []
                    joint_action = []
                    RRA_all_agents = np.zeros([len(agent_list), params.n_neighbor, 2], dtype="int32")

                    greedy_joint_action = []
                    for ag_idx in range(len(agent_list)):
                        agent_list[ag_idx].eps_threshold = 0
                        action = agent_list[ag_idx].select_action(ag_state_list[ag_idx], env)
                        greedy_joint_action.append(action.item())
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
                    tx_mask = actions != (action_dim - 1)

                    global_reward, individual_ag_rewards, V2I_throughput, done = env.step(
                        RRA_all_agents.copy(), t, interval
                    )

                    _post_empty_tx_mask = pre_empty_mask & tx_mask

                    # For NFIG, add V2I throughput to global reward
                    if params.task_type == "NFIG":
                        global_reward = global_reward + sum(V2I_throughput)

                    total_rewards += global_reward

                    if global_reward > max_joint_action_reward:
                        max_joint_action_reward = global_reward
                    if global_reward < min_joint_action_reward:
                        min_joint_action_reward = global_reward

                    ag_next_state_list = []
                    for ag_idx in range(len(agent_list)):
                        ag_next_state = env.get_state([ag_idx, 0], epsi_new, t + 1)
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

        csv_file.close()
        return episode_rewards, test_rewards