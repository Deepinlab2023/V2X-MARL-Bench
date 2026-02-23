import numpy as np
import secrets
import pandas as pd
import random
from collections import defaultdict
from typing import Optional
import sys
import scipy.stats as stats
from Benchmarkers.idql_test import IDQLtester
import torch as th


class EnvironHelper:

    def __init__(self, params):
        self.state_type = params.V2I_V2V_scenario_state_type
        self.n_pw_levels = params.num_pw_levels
        self.action_dim = params.num_actions
    
        # print("self.state_type: ", self.state_type)
        # print("self.n_pw_levels: ", self.n_pw_levels)
        # print("self.action_dim: ", self.action_dim)

    def mapping_action2RRA(self, action):

        if self.state_type == 'simplified_version':


            if action < self.action_dim - 1:
                # convert action index to power allocation and SC allocation
                SC_index = self.ag_idx # self.n_SC - self.ag_idx - 1
                Power_level_index = action % self.n_pw_levels
            else:
                SC_index = -1
                Power_level_index = -1
        else:

            if action < self.action_dim - 1:
                # convert action index to power allocation and SC allocation
                SC_index = int(np.floor(action.cpu().numpy() / self.n_pw_levels))
                Power_level_index = action % self.n_pw_levels
            else:
                SC_index = -1
                Power_level_index = -1
                
        return SC_index, Power_level_index
    
    def set_seed(seed):
            np.random.seed(seed)  # NumPy random generator
            random.seed(seed)  # Python’s built-in random module
            th.manual_seed(seed)  # PyTorch CPU
            th.cuda.manual_seed_all(seed)  # PyTorch GPU
            th.backends.cudnn.deterministic = True
            th.backends.cudnn.benchmark = False  # Ensure determinism in training


# loading vehicle postions over a time period. The time period will be greater than
# nb_episodes_control * t_max_control * n_step_per_episode_communication
# nb_episodes_control * (120 control intervals by default) * [50 comm intervals (50ms)]

def load_veh_pos(file_name):
    file_path = file_name
    data = pd.read_csv(file_path)

    return data



def random_sample(t_max_control, data):


    block_id = (data['time'] != data['time'].shift()).cumsum()
    data = data.copy()
    data['block_id'] = block_id
    blocks = data['block_id'].unique()

    if t_max_control > len(blocks):
        print("Error: not enough blocks to sample")
        sys.exit(1)

    chosen_blocks = np.random.choice(blocks, size=t_max_control, replace=False)
    sampled_data = data[data['block_id'].isin(chosen_blocks)].drop(columns='block_id')

    return sampled_data




def consecutive_sample(n_intervals: int, df: pd.DataFrame, *, seed=None) -> pd.DataFrame:


    H = int(n_intervals)
    if H <= 0:
        raise ValueError("n_intervals must be >= 1")
    if "time" not in df.columns:
        raise ValueError("DataFrame must contain a 'time' column")


    need_init = (
        not hasattr(consecutive_sample, "_state")
        or consecutive_sample._state.get("df_key") != id(df)
        or consecutive_sample._state.get("H") != H
        or consecutive_sample._state.get("schema") != tuple(df.columns)
        or consecutive_sample._state.get("nrows") != len(df)
    )

    if need_init:
        df2 = df.copy()
        df2["block_id"] = (df2["time"].diff() < 0).cumsum()

        blocks = []
        for bid, grp in df2.groupby("block_id", sort=True):
            times = np.sort(grp["time"].unique())
            if len(times) >= H:
                blocks.append({"bid": int(bid), "times": times})
        if not blocks:
            raise StopIteration("No block has at least n_intervals unique time steps.")

        rng = np.random.default_rng(seed)

        consecutive_sample._state = {
            "df_key": id(df),
            "schema": tuple(df.columns),
            "nrows": len(df),
            "df2": df2,
            "blocks": blocks,
            "rng": rng,
            "H": H,
            "k": 0,          # position within current window [0..H-1]
            "b": None,       # current block index
            "s": None,       # current window start within block
        }

    st = consecutive_sample._state
    df2, blocks, rng, H = st["df2"], st["blocks"], st["rng"], st["H"]
    k, b, s = st["k"], st["b"], st["s"]

    if k == 0:
        b = int(rng.integers(0, len(blocks)))
        times = blocks[b]["times"]
        s = int(rng.integers(0, len(times) - H + 1))
        st["b"], st["s"] = b, s


    blk = blocks[b]
    times = blk["times"]
    tval = times[s + k]


    k += 1
    if k == H:
        k = 0  # next call will pick a NEW random block/start
    st["k"] = k


    out = (
        df2[(df2["block_id"] == blk["bid"]) & (df2["time"] == tval)]
        .drop(columns="block_id")
        .sort_values(["time"], kind="stable")
        .reset_index(drop=True)
    )
    return out


# Optional helper if you ever want to force a restart manually:
def _consecutive_sample_reset_impl(df: Optional[pd.DataFrame] = None):
    if hasattr(consecutive_sample, "_state"):
        if df is None or consecutive_sample._state.get("df_key") == id(df):
            del consecutive_sample._state
consecutive_sample.reset = _consecutive_sample_reset_impl


def ordered_sample(te, data):
    """
    Return the block at position `te` (0-based), cycling when te >= #blocks.
    """
    data = data.copy()
    data['block_id'] = (data['time'] != data['time'].shift()).cumsum()
    blocks = data['block_id'].unique()          # preserves first-seen order
    idx = te % len(blocks)                       # wrap-around
    chosen_block = blocks[idx]
    return data[data['block_id'] == chosen_block].drop(columns='block_id')




# This will sample [t_max_control] number of timesteps
# Used for games with continuous control intervals
def sample_veh_positions(t_max_control, data):
    # Get the number of unique time steps in the data
    unique_time_steps = data['time'].nunique()

    # Check if there are enough time steps to sample
    if t_max_control > unique_time_steps:
        print("Error: not enough time steps to sample")
        sys.exit(1)

    # Get the unique time steps in sorted order
    sorted_time_steps = data['time'].drop_duplicates().sort_values()

    # Randomly select a starting index for sampling a block of timesteps
    start_index = np.random.randint(0, unique_time_steps - t_max_control + 1)
    # start_index = secrets.randbelow(unique_time_steps - t_max_control + 1)

    # Select the successive time steps starting from the random start index
    sampled_time_steps = sorted_time_steps.iloc[start_index:start_index + t_max_control]

    # Filter the DataFrame to include only the rows with the selected time steps
    sampled_data = data[data['time'].isin(sampled_time_steps)]

    return sampled_data


# This will sample 1 timestep
# Used for NFIG and queue-aware environments
def sample_veh_position_single(data):
    # Get the number of unique time steps in the data
    unique_time_steps = data['time'].nunique()

    # Check if there are any time steps to sample
    if unique_time_steps == 0:
        print("Error: no time steps to sample")
        sys.exit(1)

    # Get the unique time steps in sorted order
    sorted_time_steps = data['time'].drop_duplicates().sort_values()

    # Randomly select one time step
    sampled_time_step = np.random.choice(sorted_time_steps)

    # Filter the DataFrame to include only the rows with the selected time step
    sampled_data = data[data['time'] == sampled_time_step]

    return sampled_data


def sample_veh_position_from_timestep(data, time_step):
    # Check if the provided time step exists in the data
    if time_step not in data['time'].unique():
        print(f"Error: time step {time_step} not found in the data")
        return None

    # Filter the DataFrame to include only the rows with the provided time step
    sampled_data = data[data['time'] == time_step]

    return sampled_data

# Remove all test data from veh_pos_data
def remove_test_data_from_veh_pos(veh_pos_data, time_steps):
    # Filter veh_pos_data to exclude rows with time in time_steps
    filtered_data = veh_pos_data[~veh_pos_data['time'].isin(time_steps)]
    
    return filtered_data

def generate_actions_with_none(agent_idx, current_action, all_actions, action_dim, agent_number, null_ID):
    if agent_idx == agent_number:
        all_actions.append(current_action.copy())
        return
    if agent_idx in null_ID:
        # Agent 1's action is always None
        current_action[agent_idx] = None
        generate_actions_with_none(agent_idx + 1, current_action, all_actions, action_dim, agent_number, null_ID)
    else:
        for action in range(action_dim):
            current_action[agent_idx] = action
            generate_actions_with_none(agent_idx + 1, current_action, all_actions, action_dim, agent_number, null_ID)



def enumerate_all_actions_with_none(action_dim, agent_number, null_ID):
    all_actions = []
    current_action = [0] * agent_number
    generate_actions_with_none(0, current_action, all_actions, action_dim, agent_number, null_ID)
    return all_actions


def generate_actions(agent_idx, current_action, all_actions, action_dim, agent_number):
    if agent_idx == agent_number:
        all_actions.append(current_action.copy())
        return
    # Iterate over all possible actions for the current agent
    for action in range(action_dim):
        current_action[agent_idx] = action
        generate_actions(agent_idx + 1, current_action, all_actions, action_dim, agent_number)

def enumerate_all_actions(action_dim, agent_number):
    all_actions = []
    current_action = [0] * agent_number
    generate_actions(0, current_action, all_actions, action_dim, agent_number)
    return all_actions



def calculate_max_mean_and_ci(data, confidence=0.95):
    """
    Calculate the maximum mean result of any time step and the corresponding confidence interval.
    Also returns the mean and confidence interval over time for all time steps.
    
    Parameters:
    - data: 2D list or array where each row represents a different run of the experiment and each column represents a time step.
    - confidence: Confidence level for the confidence interval (default is 0.95).
    
    Returns:
    - max_mean: Maximum mean result at any time step.
    - max_mean_ci: Confidence interval for the maximum mean result.
    - mean_over_time: Mean result for each time step across all runs.
    - ci_over_time: Confidence interval for each time step across all runs.
    """
    # Convert the data to a NumPy array for easier processing
    data = np.array(data)
    
    # Calculate the mean across runs for each time step
    mean_over_time = np.mean(data, axis=0)
    
    # Find the index of the time step with the maximum mean
    max_mean_index = np.argmax(mean_over_time)
    
    # Extract the data corresponding to the max mean time step
    max_mean_data = data[:, max_mean_index]
    
    # Calculate the mean and standard error for the max mean time step
    max_mean = np.mean(max_mean_data)
    std_error = stats.sem(max_mean_data)
    
    # Calculate the confidence interval for the max mean time step
    max_mean_ci = std_error * stats.t.ppf((1 + confidence) / 2, len(max_mean_data) - 1)
    
    # Calculate confidence intervals over time for all time steps
    std_error_over_time = stats.sem(data, axis=0)
    ci_over_time = std_error_over_time * stats.t.ppf((1 + confidence) / 2, data.shape[0] - 1)
    
    return max_mean, max_mean_ci, mean_over_time, ci_over_time


def average_reward_for_agent_action(reward_dict, agent_index, action_index):

    total_reward = 0.0
    count = 0

    for joint_action, reward in reward_dict.items():
        if joint_action[agent_index] == action_index:
            total_reward += reward
            count += 1

    if count == 0:
        return 0.0

    return total_reward / count

def select_max_joint_action(reward_dict, num_agents, action_dim):
    max_joint_action = []
    max_avg_rewards = []

    for agent_index in range(num_agents):
        max_action = None
        max_avg = float('-inf')

        for action_index in range(action_dim):
            avg_reward = average_reward_for_agent_action(reward_dict, agent_index, action_index)

            if avg_reward > max_avg:
                max_avg = avg_reward
                max_action = action_index

        max_joint_action.append(max_action)
        max_avg_rewards.append(max_avg)

    return max_joint_action, max_avg_rewards

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kstest, normaltest, skew, kurtosis

def plot_joint_action_distribution(joint_action_dic,
                                   bins=40, kde=True, show_stats=True,
                                   figsize=(6,4)):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde, kstest, normaltest, skew, kurtosis

    # ---- 1-D array of scalars  ----
    rewards = np.array([float(v) for v in joint_action_dic.values()])

    plt.figure(figsize=figsize)
    counts, edges, _ = plt.hist(rewards, bins=bins, density=True,
                                alpha=0.4, edgecolor="k", label="histogram")

    if kde:
        xx = np.linspace(rewards.min(), rewards.max(), 500)
        plt.plot(xx, gaussian_kde(rewards)(xx), lw=2, label="KDE")

    plt.xlabel("global reward"); plt.ylabel("density")
    plt.title("Distribution of joint-action global rewards")
    plt.legend(); plt.tight_layout(); plt.show()

    if show_stats:
        print("count =", len(rewards))
        print("mean  =", rewards.mean())
        print("std   =", rewards.std(ddof=1))
        print("skew  =", skew(rewards))
        print("kurtosis (excess) =", kurtosis(rewards))
        print("KS p-value :", kstest((rewards-rewards.mean())/rewards.std(ddof=1),"norm").pvalue)
        print("D’Agostino p-value :", normaltest(rewards).pvalue)

# ---------- generic helper to plot one 1-D distribution ----------
def _plot_distribution(samples, title, xlabel, bins=40, kde=True, show_stats=True, figsize=(6,4)):
    plt.figure(figsize=figsize)
    counts, edges, _ = plt.hist(samples, bins=bins, density=True,
                                alpha=0.4, edgecolor="k", label="histogram")
    if kde and len(samples) > 1:
        xx = np.linspace(min(samples), max(samples), 500)
        plt.plot(xx, gaussian_kde(samples)(xx), lw=2, label="KDE")

    plt.xlabel(xlabel); plt.ylabel("density")
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

    if show_stats:
        samples = np.asarray(samples)
        print("count =", len(samples))
        print("mean  =", samples.mean())
        print("std   =", samples.std(ddof=1))
        print("skew  =", skew(samples))
        print("kurtosis (excess) =", kurtosis(samples))
        print("KS p-value :", kstest((samples-samples.mean())/samples.std(ddof=1), "norm").pvalue)
        print("D’Agostino p-value :", normaltest(samples).pvalue)



# ---------- convenience wrappers with sample count in the title ----------
def plot_interference_for_agent_action(interference_dict,
                                       agent_idx, action,
                                       **plot_kwargs):
    key = (agent_idx, action)
    samples = interference_dict.get(key)
    if samples is None:
        print(f"No samples recorded for agent {agent_idx} taking action {action}")
        return

    n = len(samples)
    _plot_distribution(samples,
                       title = f"Interference (dBm) | agent {agent_idx}, "
                               f"action {action}  (n={n})",
                       xlabel = "interference power [dBm]",
                       **plot_kwargs)


def plot_reward_for_agent_action(reward_dict,
                                 agent_idx, action,
                                 **plot_kwargs):
    key = (agent_idx, action)
    samples = reward_dict.get(key)
    if samples is None:
        print(f"No samples recorded for agent {agent_idx} taking action {action}")
        return

    n = len(samples)
    _plot_distribution(samples,
                       title = f"Individual reward | agent {agent_idx}, "
                               f"action {action}  (n={n})",
                       xlabel = "reward",
                       **plot_kwargs)

# ---------- NEW: aggregate-over-actions helper ----------------------
def _gather_samples_for_agent(store, agent_idx):
    """
    `store` is either `reward_dict` or `interference_dict`
           produced by brute_force_joint_action.
    Returns a flat list containing one sample per *joint action*
    for the chosen agent, independent of which action that agent took.
    """
    samples = []
    for (ag, act), lst in store.items():   # loop over all buckets
        if ag == agent_idx:
            samples.extend(lst)            # concatenate
    return samples


# ---------- wrappers that ignore the agent’s action ----------------
def plot_reward_for_agent(reward_dict,
                          agent_idx,
                          **plot_kwargs):
    samples = _gather_samples_for_agent(reward_dict, agent_idx)
    if not samples:
        print(f"No reward samples recorded for agent {agent_idx}")
        return

    n = len(samples)
    _plot_distribution(samples,
        title = f"Individual reward | agent {agent_idx}  (n={n})",
        xlabel = "reward",
        **plot_kwargs)


def plot_interference_for_agent(interference_dict,
                                agent_idx,
                                **plot_kwargs):
    samples = _gather_samples_for_agent(interference_dict, agent_idx)
    if not samples:
        print(f"No interference samples recorded for agent {agent_idx}")
        return

    n = len(samples)
    _plot_distribution(samples,
        title = f"Interference (dBm) | agent {agent_idx}  (n={n})",
        xlabel = "interference power [dBm]",
        **plot_kwargs)
