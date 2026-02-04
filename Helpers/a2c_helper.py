import csv
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from typing import Optional

from Environment.environment_utility import (
    sample_veh_position_from_timestep,
    random_sample,
    consecutive_sample,
    ordered_sample,
)

device = th.device("cuda" if th.cuda.is_available() else "cpu")


class A2CHelper:
    def __init__(self):
        pass

    # ---------- feature suffix for filenames ----------
    @staticmethod
    def build_feature_suffix(params):
        tags = []

        if getattr(params, "action_masking", False):
            tags.append("MASK")
        if getattr(params, "adv_normalization", False):
            tags.append("NORM")

        return "_" + "_".join(tags) if tags else ""

    # ---------- generic A2C compat checks for IA2C / MAA2C ----------
    @staticmethod
    def basic_a2c_compat_checks(params, algo_name: str):
        p = params

        # POSIG + no_sharing is illegal
        if getattr(p, "no_sharing", False) and p.task_type == "POSIG":
            raise ValueError(
                f"{algo_name} No-Sharing is not supported for POSIG. "
                "Set params.no_sharing = False for POSIG experiments."
            )

    # ---------- CSV logging / filename construction ----------
    @staticmethod
    def init_csv_logging(params, algo_name: str, posig_tag: Optional[str] = None):

        suffix = A2CHelper.build_feature_suffix(params)

        prefix = (
            f"{algo_name}NS_trial_"
            if getattr(params, "no_sharing", False)
            else f"{algo_name}_trial_"
        )

        if params.task_type == "NFIG":
            csv_name = (
                f"{prefix}{params.trial_run}_NFIG"
                f"{int(params.n_agent)}{int(params.n_sc)}_"
                f"{int(params.loc)}{suffix}.csv"
            )

        elif params.task_type == "SIG":
            if params.loc is None:
                csv_name = (
                    f"{prefix}{params.trial_run}_SIG"
                    f"{int(params.n_agent)}{int(params.n_sc)}_"
                    f"ML_{params.fast_fading_tag}{suffix}.csv"
                )
            else:
                csv_name = (
                    f"{prefix}{params.trial_run}_SIG"
                    f"{int(params.n_agent)}{int(params.n_sc)}_"
                    f"SL_{params.fast_fading_tag}_{params.loc}{suffix}.csv"
                )

        elif params.task_type == "POSIG":
            if posig_tag:
                csv_name = f"{prefix}{params.trial_run}_POSIG_{posig_tag}{suffix}.csv"
            else:
                csv_name = f"{prefix}{params.trial_run}_POSIG{suffix}.csv"
        else:
            raise ValueError(f"Unknown task_type: {params.task_type}")

        csv_file = open(csv_name, "a", newline="")
        csv_writer = csv.writer(csv_file)
        return csv_file, csv_writer

    # ---------- shared control-interval logic ----------
    @staticmethod
    def num_control_intervals(params) -> int:
        # All task types now use single control interval
        # (t_max_control is obsolete in refactored environment)
        return 1

    # ---------- shared veh-position sampling ----------
    @staticmethod
    def sample_veh_positions(params):
        p = params

        if p.task_type == "NFIG" and p.loc is not None:
            return sample_veh_position_from_timestep(p.train_data, p.loc)

        elif (p.task_type in ["SIG", "POSIG"]) and p.loc is None:
            # if p.sample_method == "random":
            #     return random_sample(p.sampling_size, p.veh_pos_data)
            # elif p.sample_method == "consecutive":
            #     return consecutive_sample(p.sampling_size, p.veh_pos_data)
            # elif p.sample_method == "ordered":
            #     return ordered_sample(p.sampling_size, p.veh_pos_data)
            # else:
            #     raise ValueError(f"Unknown sample_method: {p.sample_method}")
            return random_sample(1, p.train_data)

        elif p.task_type == "SIG" and p.loc is not None:
            return sample_veh_position_from_timestep(p.train_data, p.loc)

        else:
            raise ValueError(f"Unsupported task_type/loc combination: {p.task_type}, {p.loc}")

    # ---------- return computation ----------
    @staticmethod
    def compute_returns_from_buffer(buffer, done: bool, gamma: float):
        if not done:
            raise ValueError(
                "Episode Terminated Unexpectedly! Done flag not reached "
                "(expected done=True at final timestep)."
            )

        R = 0.0
        rtrns = []
        for data in reversed(buffer):
            *_, reward = data
            R = reward + gamma * R
            rtrns.append(float(R))

        rtrns.reverse()
        return th.tensor(np.array(rtrns), dtype=th.float32, device=device)

    # ---------- batch building (FO vs POSIG, IA2C vs MAA2C) ----------
    @staticmethod
    def build_batches(params, batch_buffer, batch_rtrns, ctde: bool):
        batch_training = A2CBatchTraining()
        task_type = params.task_type

        if task_type == "POSIG":
            if ctde:
                (batch_global_state, batch_observations, batch_joint_actions, batch_rtrns_tensor) = \
                    batch_training.collate_batch(batch_buffer, batch_rtrns, task_type, ctde=True)
                has_obs = True
            else:
                (batch_observations, batch_joint_actions, batch_rtrns_tensor) = \
                    batch_training.collate_batch(batch_buffer, batch_rtrns, task_type, ctde=False)
                batch_global_state = None
                has_obs = True
        else:
            (batch_global_state, batch_joint_actions, batch_rtrns_tensor) = \
                batch_training.collate_batch(batch_buffer, batch_rtrns, task_type, ctde=ctde)
            batch_observations = None
            has_obs = False

        return batch_global_state, batch_observations, batch_joint_actions, batch_rtrns_tensor, has_obs

    # ---------- single training iteration ----------
    @staticmethod
    def run_training_iteration(trainer, ctde: bool):
        p = trainer.params
        batch_buffer = []
        batch_rtrns = []

        for _ in range(p.batch_size):
            buffer, rtrns, ep_reward = trainer._run_single_episode()
            trainer.episode_rewards.append(float(ep_reward))
            trainer.episode += 1

            batch_buffer.extend(buffer)
            batch_rtrns.extend(rtrns)

            trainer._run_test()

        (batch_global_state, batch_observations, batch_joint_actions, batch_rtrns_tensor, has_obs) = \
            A2CHelper.build_batches(p, batch_buffer, batch_rtrns, ctde)

        trainer._update_from_batch(
            batch_global_state,
            batch_observations,
            batch_joint_actions,
            batch_rtrns_tensor,
            has_obs,
        )

    # ---------- full train loop ----------
    @staticmethod
    def train_loop(trainer, ctde: bool):
        p = trainer.params
        for _ in range(p.num_training_iterations):
            A2CHelper.run_training_iteration(trainer, ctde)
        return trainer.episode_rewards, trainer.test_rewards

    # ---------- common test logging / plotting ----------
    @staticmethod
    def finalize_test(trainer, test_reward: float, algo_name: str):
        trainer.test_rewards.append(test_reward)
        trainer.csv_writer.writerow([test_reward])
        trainer.csv_file.flush()

        plt.figure(1)
        plt.clf()
        plt.plot(trainer.test_rewards, label="Test Reward")
        plt.xlabel("Test Interval")
        plt.ylabel("Return")
        plt.title(f"Test Return Over Time {algo_name}")
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(1)

        print(f"Training reward at episode {trainer.episode}: {test_reward:.2f}")


class A2CBatchTraining:
    def __init__(self):
        pass

    def collate_batch(self, buffer, rtrns, task_type, ctde: bool):
        batch_joint_actions = []
        batch_rtrns_list = []
        batch_global_states = []
        batch_observations = []

        for (data, R) in zip(buffer, rtrns):
            if task_type == "POSIG":
                if ctde:
                    global_state, observations, joint_action, _ = data
                    batch_global_states.append(global_state)
                else:
                    observations, joint_action, _ = data

                obs_tensor = th.stack(observations, dim=0)  # [A, obs_dim]
                batch_observations.append(obs_tensor)
            else:
                global_state, joint_action, _ = data
                batch_global_states.append(global_state)

            batch_joint_actions.append(joint_action)
            batch_rtrns_list.append(R)

        batch_joint_actions = th.tensor(batch_joint_actions, dtype=th.long, device=device)  # [N, A]
        batch_rtrns = th.tensor(batch_rtrns_list, dtype=th.float32, device=device)         # [N]

        if task_type == "POSIG":
            batch_observations = th.stack(batch_observations).to(device)  # [N, A, obs_dim]
            if ctde:
                batch_global_states = th.stack(batch_global_states).to(device)  # [N, state_dim]
                return batch_global_states, batch_observations, batch_joint_actions, batch_rtrns
            else:
                return batch_observations, batch_joint_actions, batch_rtrns
        else:
            batch_global_states = th.stack(batch_global_states).to(device)
            return batch_global_states, batch_joint_actions, batch_rtrns