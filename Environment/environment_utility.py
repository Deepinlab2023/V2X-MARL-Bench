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




# def consecutive_sample(n_intervals: int, df: pd.DataFrame) -> pd.DataFrame:

#     # 1. Identify blocks (= individual SUMO simulations)
#     df = df.copy()
#     df["block_id"] = (df["time"].diff() < 0).cumsum()  # time jumps → new block

#     # 2. Keep only blocks long enough
#     candidates = []
#     for bid, grp in df.groupby("block_id"):
#         unique_times = grp["time"].unique()
#         if len(unique_times) >= n_intervals:
#             candidates.append((bid, np.sort(unique_times)))

#     if not candidates:
#         raise ValueError("No block has enough consecutive time-steps")

#     # 3. Pick one block at random, then pick a random start index
#     bid, times = random.choice(candidates)
#     start = random.randint(0, len(times) - n_intervals)
#     chosen_times = times[start : start + n_intervals]

#     # 4. Return rows that match those time-steps in that block
#     sample = df[(df["block_id"] == bid) & (df["time"].isin(chosen_times))].drop(columns="block_id")
#     return sample


# import random
# from typing import Optional
# import numpy as np
# import pandas as pd



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


# --- only two extra imports -----------------------------------------------
import secrets, numpy as np             #  np was already used lower down

def monte_carlo_joint_action(params, env,
                              num_agents, action_dim, agent_list,
                              n_samples=5_000_000, seed=None,
                              collect_agent_stats=True):
    """
    Randomly sample `n_samples` joint actions (true-random seed by default)
    and reuse the original plotting / logging logic.
    """
    from itertools import product   # <-- still needed for mapping loop
    import torch
    from collections import defaultdict

    # ---------------------- RNG seeded from OS entropy ---------------------
    if seed is None:
        seed = int.from_bytes(secrets.token_bytes(16), "big")
    rng = np.random.default_rng(seed)
    print(f"[INFO] RNG seeded with {seed}")

    environ_helper = EnvironHelper(params)

    optimal_joint_action = None
    optimal_RRA = None
    optimal_joint_reward = -np.inf
    joint_action_dic   = {}

    num_sc          = 8            # <-- 3 in your 12ag-3SC case
    max_agents      = num_agents               # 12 here
    occ_counter     = np.zeros(max_agents+1, dtype=np.int64)  # one bin per m
    tot_sc_samples  = 0                        # total (#MC samples × #SC)

    interference_dict  = defaultdict(list)
    individual_reward_dict = defaultdict(list)
    eps = 1e-10

    # -----------------------  Monte-Carlo loop ----------------------------
    for _ in range(n_samples):
        ja = rng.integers(0, action_dim, size=num_agents).tolist()

        # ------------------------------------------------------------------
        active = np.array(ja)            # shape (num_agents,)
        non_silent_mask = (active != 9)  # 9 = silent
        active = active[non_silent_mask] # drop silent actions

        # --- map action ∈ {0..8} → channel index ∈ {0,1,2}
        if active.size > 0:
            ch_idx = active // 3         # integer division
            counts = np.bincount(ch_idx, minlength=num_sc)
        else:
            counts = np.zeros(num_sc, dtype=int)

        # update histogram
        for c in counts:                 # c = #non-silent users on this SC
            occ_counter[c] += 1
        tot_sc_samples += num_sc
        # ------------------------------------------------------------------


        RRA_all_agents = np.zeros([num_agents, params.n_neighbor, 2], dtype='int32')
        joint_action   = []
        for ag_idx, act in enumerate(ja):
            action = torch.tensor([[act]])
            joint_action.append(action)


            RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = environ_helper.mapping_action2RRA(action)




        V2V_throughput, indiv_ag_rewards, V2I_throughput, done, tot_V2V_Interf = \
            env.step(RRA_all_agents.copy(), 0, 1)

        global_reward = V2V_throughput + sum(V2I_throughput)
        joint_action_dic[tuple(ja)] = global_reward

        # ---------- stats collection (unchanged) -------------------------
        if collect_agent_stats:
            for ag_idx, act in enumerate(ja):
                interf_mW = float(tot_V2V_Interf[ag_idx, 0]) + eps
                interference_dict[(ag_idx, act)].append(10*np.log10(interf_mW))
                individual_reward_dict[(ag_idx, act)].append(float(indiv_ag_rewards[ag_idx, 0]))

        # ---------- running optimum (unchanged) --------------------------
        if global_reward > optimal_joint_reward:
            optimal_joint_reward = global_reward
            optimal_joint_action = ja
            optimal_RRA          = RRA_all_agents
            optimal_V2V          = V2V_throughput
            optimal_V2I          = V2I_throughput


    # --------------- after the Monte-Carlo loop finishes ---------------
    print("\nProbability mass function for 'm non-silent users on an SC'")
    for m, freq in enumerate(occ_counter):
        prob = freq / tot_sc_samples
        print(f"P{{occupancy = {m:2d}}} ≈ {prob:.6f}")

    # -----------------------  summary printout ----------------------------
    print("samples_evaluated     :", n_samples)
    print("optimal_V2V / V2I     :", optimal_V2V, optimal_V2I)
    print("optimal_joint_reward  :", optimal_joint_reward)
    print("optimal_joint_action  :", optimal_joint_action)
    print("optimal_RRA           :", optimal_RRA)
    print("================================================")

    # -------------- reuse your existing plotting function -----------------
    plot_joint_action_distribution(joint_action_dic,
                                   bins=50, kde=True)  # <- same call

    return optimal_joint_action, optimal_joint_reward


def brute_force_joint_action(params, env, num_agents, action_dim, agent_list):

    from itertools import product
    import torch

    total_joint_action_num = 0
    environ_helper = EnvironHelper(params)

    joint_actions = [list(a) for a in product(range(action_dim), repeat=num_agents)]


    optimal_joint_action = None
    optimal_RRA = None
    optimal_joint_reward = 0
    
    joint_action_dic = {}
    interference_dict = defaultdict(list)   # key = (agent_idx, action)
    individual_reward_dict = defaultdict(list)   # key = (agent_idx, action)
    eps = 1e-10

    for ja in joint_actions:
        RRA_all_agents = np.zeros([num_agents, params.n_neighbor, 2], dtype='int32')
        joint_action = []
        total_joint_action_num += 1
        for ag_idx in range(num_agents):
            action = torch.tensor([[ja[ag_idx]]])
            joint_action.append(action)


            RRA_all_agents[ag_idx, 0, 0], RRA_all_agents[ag_idx, 0, 1] = environ_helper.mapping_action2RRA(action)


        # V2V_throughput, individual_ag_rewards, V2I_throughput, done = env.step(RRA_all_agents.copy(), 0, 1)
        V2V_throughput, individual_ag_rewards, V2I_throughput, done = env.step(RRA_all_agents.copy(), 0, 1)
        # global_reward = V2V_throughput + sum(V2I_throughput)
        global_reward = V2V_throughput + sum(V2I_throughput)

        # global_reward = V2V_throughput
        joint_action_dic[tuple(ja)] = global_reward

        is_NE, _ = IDQLtester.is_pure_NE(env, joint_action, global_reward, agent_list[0].action_dim, environ_helper, agent_list, params)
        if is_NE:
            print("RRA_all_agents: ", RRA_all_agents)
            print("global_reward: ", global_reward)
        # if global_reward > 117.0:
        #     print(f"RRA_all_agents: {RRA_all_agents} v2v_se: {V2V_throughput} v2i_se: {V2I_throughput} individual_ag_rewards: {individual_ag_rewards} global_reward: {global_reward}")

        # print("tot_V2V_Interference: ")
        # print(tot_V2V_Interference)
        # print("individual_ag_rewards: ")
        # print(individual_ag_rewards)
        # print("global_reward: ")
        # print(global_reward)

        collect_agent_stats = True
        # ---- LOG conditional samples (only if wanted) ---------------
        if collect_agent_stats:
            for ag_idx in range(num_agents):
                act = ja[ag_idx]

                # interference at *this* agent’s receiver; convert → dBm
                # interf_mW = float(tot_V2V_Interference[ag_idx, 0]) + eps
                # interf_dBm = 10*np.log10(interf_mW)
                # interference_dict[(ag_idx, act)].append(interf_dBm)

                # agent's own reward
                individual_reward_dict[(ag_idx, act)].append(float(individual_ag_rewards[ag_idx, 0]))


        if global_reward > optimal_joint_reward:
            optimal_joint_reward = global_reward
            optimal_joint_action = ja
            optimal_RRA = RRA_all_agents
            
            # optimal_V2V = V2V_throughput
            # optimal_V2I = V2I_throughput

            optimal_V2V = V2V_throughput
            optimal_V2I = V2I_throughput



    print("total_joint_action_num: ", total_joint_action_num)
    print("optimal_V2V: ", optimal_V2V)
    print("optimal_V2I: ", optimal_V2I)
    print("optimal_joint_reward: ", optimal_joint_reward)
    print("optimal_joint_action: ", optimal_joint_action)
    print("optimal_RRA: ", optimal_RRA)

    print("============================================")

    # print("[12, 3, 9, 3] reward: ", joint_action_dic[(12, 9, 3, 9)])


    optimal_joint_action_average = []
    for ag_idx in range(num_agents):
        optimal_joint_action_average.append(average_reward_for_agent_action(joint_action_dic, ag_idx, optimal_joint_action[ag_idx]))

    max_joint_action, max_avg_rewards = select_max_joint_action(joint_action_dic, num_agents, action_dim)

    print("optimal_joint_action_average: ",optimal_joint_action, optimal_joint_action_average)
    print("max_joint_action_average", max_joint_action, max_avg_rewards)

    print("============================================")

    # print(joint_action_dic)
    plot_joint_action_distribution(joint_action_dic,
                                   bins=50,   # finer resolution
                                   kde=True)  # smooth overlay

    # # Example: agent 0 chooses action 5  →  interference distribution
    # plot_interference_for_agent_action(interference_dict, agent_idx=0, action=5,
    #                                    bins=50, kde=True)

    # # # Example: same condition but plot that agent’s reward distribution
    # plot_reward_for_agent_action(individual_reward_dict, agent_idx=4, action=0,
    #                              bins=130, kde=True)
    # plot_reward_for_agent_action(individual_reward_dict, agent_idx=4, action=3,
    #                             bins=130, kde=True)



    # plot_reward_for_agent(individual_reward_dict, agent_idx=0, bins=50, kde=True)
    # plot_reward_for_agent(individual_reward_dict, agent_idx=2, bins=50, kde=True)
    # plot_reward_for_agent(individual_reward_dict, agent_idx=4, bins=50, kde=True)
    # plot_reward_for_agent(individual_reward_dict, agent_idx=6, bins=50, kde=True)

    # plot_interference_for_agent(interference_dict, agent_idx=0, bins=50, kde=True)
    # plot_interference_for_agent(interference_dict, agent_idx=2, bins=50, kde=True)
    # plot_interference_for_agent(interference_dict, agent_idx=4, bins=50, kde=True)
    # plot_interference_for_agent(interference_dict, agent_idx=6, bins=50, kde=True)




    # plot_reward_for_agent_action(individual_reward_dict, agent_idx=0, action=5, bins=40, kde=True)


    # print("converged average: ", average_reward_for_agent_action(joint_action_dic, 0, 1), average_reward_for_agent_action(joint_action_dic, 1, 6),
    #                             average_reward_for_agent_action(joint_action_dic, 2, 6), average_reward_for_agent_action(joint_action_dic, 3, 3))


    # RRA_all_agents = np.zeros([num_agents, params.n_neighbor, 2], dtype='int32')
    # RRA_all_agents[0, 0, 0], RRA_all_agents[0, 0, 1] = 1, 0
    # RRA_all_agents[1, 0, 0], RRA_all_agents[1, 0, 1] = -1, -1
    # RRA_all_agents[2, 0, 0], RRA_all_agents[2, 0, 1] = 0, 0
    # RRA_all_agents[3, 0, 0], RRA_all_agents[3, 0, 1] = -1, -1
    # global_reward, individual_ag_rewards, V2I_throughput, done = env.step(RRA_all_agents.copy(), 0, 1)
    # global_reward = global_reward + sum(V2I_throughput)
    # print("RRA_all_agents: ", RRA_all_agents)
    # print("global_reward: ", global_reward)


    return optimal_joint_action, optimal_joint_reward

import matplotlib.pyplot as plt
from collections import Counter

def plot_dict_value_distribution(data_dict, title="Value Distribution of Dictionary"):

    value_counts = Counter(data_dict.values())
    sorted_items = sorted(value_counts.items())
    
    x = [val for val, count in sorted_items]
    y = [count for val, count in sorted_items]

    # Convert x-axis to strings to treat them as categories
    x_labels = [str(val) for val in x]

    plt.figure(figsize=(8, 5))
    plt.bar(x_labels, y, width=0.8)
    plt.xlabel("Value in dictionary")
    plt.ylabel("Number of keys with this value")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def plot_joint_action_prob(
    joint_action_list, 
    num_agents=4, 
    num_sub_channels=2, 
    power_levels=3, 
    window_size=100,
    stride=100
):
    """
    Plots joint action pattern probabilities over time using a moving window.

    Parameters:
    - joint_action_list: List of joint actions (e.g., [[0,1,4,6], ...])
    - num_agents: Number of agents in each joint action
    - num_sub_channels: Number of sub-channels (used to define action sets)
    - power_levels: Number of power levels per sub-channel
    - window_size: How many episodes to look back when computing local stats
    - stride: Plot every `stride` episodes (e.g., every 1000)
    """
    not_transmit_action = num_sub_channels * power_levels
    ch1_set = set(range(0, power_levels))
    ch2_set = set(range(power_levels, 2 * power_levels))

    total_episodes = len(joint_action_list)
    steps = range(window_size, total_episodes + 1, stride)

    probs_all_not = []
    probs_one_transmit = []
    probs_two_distinct = []

    for i in steps:
        window = joint_action_list[i - window_size:i]

        count_pattern1 = 0
        count_pattern2 = 0
        count_pattern3 = 0

        for joint_action in window:
            non_transmit_count = sum([a == not_transmit_action for a in joint_action])
            transmit_agents = [idx for idx, a in enumerate(joint_action) if a != not_transmit_action]

            if non_transmit_count == num_agents:
                count_pattern1 += 1
            elif len(transmit_agents) == 1:
                count_pattern2 += 1
            elif len(transmit_agents) == 2:
                a1 = joint_action[transmit_agents[0]]
                a2 = joint_action[transmit_agents[1]]
                if (a1 in ch1_set and a2 in ch2_set) or (a1 in ch2_set and a2 in ch1_set):
                    count_pattern3 += 1

        total = window_size
        probs_all_not.append(count_pattern1 / total)
        probs_one_transmit.append(count_pattern2 / total)
        probs_two_distinct.append(count_pattern3 / total)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(steps, probs_all_not, label='All Not Transmit')
    plt.plot(steps, probs_one_transmit, label='One Agent Transmits')
    plt.plot(steps, probs_two_distinct, label='Two Agents on Distinct Channels')

    plt.xlabel(f"Training Episode (window={window_size})")
    plt.ylabel("Probability")
    plt.title("Joint Action Pattern Probabilities Over Time")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# ============================SIG SL optimal policy search ================================

import copy
from math import ceil
from typing import List, Tuple, Optional
import numpy as np
import torch

def plan_SIG_SL_greedy(
    params,
    env,
    num_agents: int,
    action_dim: int,
    environ_helper,
    *,
    horizon: int = 50,                 # self.n_step_per_episode_communication
    game_mode: int = 2,                # we read env.global_reward from step()[0]
    silent_action_idx: Optional[int] = None,   # default: last action id
    queue_round_decimals: int = 12,    # used only for stable keys (lightweight)
    top_k_per_agent: int = 5,          # candidate actions per agent (small = faster)
    local_search_passes: int = 1       # 0..2; 1 is usually enough
) -> Tuple[List[List[int]], List[float], float]:
    """
    Fast near-optimal greedy planner for SIG-SL.

    Returns:
      joint_actions_pre_empty : list of joint action-id lists from t=0 until all queues empty
      per_step_rewards_full   : list of length = horizon with env global_reward each step
      total_return_full       : sum(per_step_rewards_full)
    """
    assert game_mode == 2, "This planner assumes game_mode == 2 reward semantics."

    # ---------- helpers ----------
    def flatten_q(qarr) -> np.ndarray:
        return np.array(qarr, dtype=float).reshape(-1)

    def queues_empty(qarr) -> bool:
        return np.all(flatten_q(qarr) <= 0.0)

    if silent_action_idx is None:
        silent_action_idx = action_dim - 1  # your contract: last action is SILENT

    # Map each action-id → (subchannel, power) once
    action_to_rra: List[Tuple[int, int]] = []
    discovered_silent = None
    for a in range(action_dim):
        sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[a]]))
        sc, pw = int(sc), int(pw)
        action_to_rra.append((sc, pw))
        if (sc, pw) == (-1, -1):
            discovered_silent = a
    # Prefer explicit mapping if available
    if discovered_silent is not None:
        silent_action_idx = discovered_silent

    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)

    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA

    # ---------- Prepass: solo service per agent×action (others SILENT) ----------
    base_env = copy.deepcopy(env)
    base_q = flatten_q(base_env.queue)
    per_action_service = [np.zeros(action_dim, dtype=float) for _ in range(num_agents)]
    per_agent_best = np.zeros(num_agents, dtype=float)

    for ag in range(num_agents):
        best = 0.0
        silent_joint = [silent_action_idx] * num_agents
        for a in range(action_dim):
            joint = silent_joint.copy()
            joint[ag] = a
            tmp = copy.deepcopy(base_env)
            RRA = rra_from_joint_actions(joint)
            # step once; we only need queue delta for this ag
            gr, _, _, _ = tmp.step(RRA.copy(), 0, 1)
            after_q = flatten_q(tmp.queue)
            served = float(max(0.0, base_q[ag] - after_q[ag]))
            per_action_service[ag][a] = served
            if served > best:
                best = served
        per_agent_best[ag] = best

    # Reduce duplicates: keep, for each agent, the best action PER SUBCHANNEL (power folded)
    best_action_per_sc: List[dict] = []
    for ag in range(num_agents):
        per_sc = {}
        for a in range(action_dim):
            sc, _ = action_to_rra[a]
            if sc == -1:  # SILENT
                continue
            val = per_action_service[ag][a]
            if sc not in per_sc or val > per_sc[sc][1]:
                per_sc[sc] = (a, val)
        best_action_per_sc.append(per_sc)  # {sc: (best_action, best_service)}

    def candidate_actions_for_agent(ag: int, q_i: float) -> List[int]:
        """Finish-capable first, then higher solo service; SILENT appended (low rank)."""
        if q_i <= 0.0:
            return [silent_action_idx]
        # build from best action per subchannel
        cands = [(a, v) for (a, v) in [best_action_per_sc[ag][sc] for sc in best_action_per_sc[ag]]]
        # add the single best among any remaining actions that weren’t per-sc winners (optional)
        # rank
        finish_flag = [(a, int(v + 1e-12 >= q_i), v) for (a, v) in cands]
        finish_flag.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ranked = [a for (a, _, _) in finish_flag][:top_k_per_agent]
        if silent_action_idx not in ranked:
            ranked.append(silent_action_idx)
        return ranked

    # ---------- Per-step greedy assignment + small local search ----------
    # Outputs
    joint_actions_pre_empty: List[List[int]] = []
    per_step_rewards_full: List[float] = []

    run_env = copy.deepcopy(env)

    for t in range(horizon):
        qnow = flatten_q(run_env.queue)

        if queues_empty(qnow):
            # After empty: everyone SILENT; keep logging rewards through horizon
            actions = [silent_action_idx] * num_agents
            RRA = rra_from_joint_actions(actions)
            global_reward, _, _, _ = run_env.step(RRA.copy(), t, 1)
            per_step_rewards_full.append(float(np.array(global_reward).reshape(-1)[0]))
            continue

        # 1) Build per-agent candidate lists
        cand_lists: List[List[int]] = []
        for ag in range(num_agents):
            cand_lists.append(candidate_actions_for_agent(ag, qnow[ag]))

        # 2) Channel-aware greedy initialization (finishers first)
        # Order agents: (finisher_available desc, q_i / best_service asc)
        order = []
        for ag in range(num_agents):
            q_i = qnow[ag]
            finisher = False
            if q_i > 0.0:
                for a in cand_lists[ag]:
                    if a == silent_action_idx: 
                        continue
                    if per_action_service[ag][a] + 1e-12 >= q_i:
                        finisher = True
                        break
            ratio = (q_i / (per_agent_best[ag] + 1e-12)) if q_i > 0 else 0.0
            order.append((ag, int(finisher), -ratio))   # finisher first, then smaller ratio
        order.sort(key=lambda x: (x[1], x[2]), reverse=True)

        assigned = [silent_action_idx] * num_agents
        used_sc = set()
        for (ag, _, _) in order:
            if qnow[ag] <= 0.0:
                assigned[ag] = silent_action_idx
                continue
            chosen = None
            # prefer a candidate on an unused subchannel
            for a in cand_lists[ag]:
                if a == silent_action_idx:
                    continue
                sc, _ = action_to_rra[a]
                if sc == -1:
                    continue
                if sc not in used_sc:
                    chosen = a
                    used_sc.add(sc)
                    break
            # if all good candidates collide, take the top one anyway
            if chosen is None:
                for a in cand_lists[ag]:
                    if a != silent_action_idx:
                        chosen = a
                        used_sc.add(action_to_rra[a][0])
                        break
            if chosen is None:
                chosen = silent_action_idx
            assigned[ag] = chosen

        # 3) Small coordinate-ascent (local improvement)
        if local_search_passes > 0:
            for _ in range(local_search_passes):
                improved = False
                # Evaluate current joint
                base_env = copy.deepcopy(run_env)
                RRA_cur = rra_from_joint_actions(assigned)
                gr_cur, _, _, _ = base_env.step(RRA_cur.copy(), t, 1)
                best_val = float(np.array(gr_cur).reshape(-1)[0])

                # Try per-agent swaps
                for ag in range(num_agents):
                    if qnow[ag] <= 0.0:
                        continue
                    best_a = assigned[ag]
                    for a in cand_lists[ag]:
                        if a == assigned[ag]:
                            continue
                        trial = assigned.copy()
                        trial[ag] = a
                        tmp_env = copy.deepcopy(run_env)
                        RRA_try = rra_from_joint_actions(trial)
                        gr_try, _, _, _ = tmp_env.step(RRA_try.copy(), t, 1)
                        val = float(np.array(gr_try).reshape(-1)[0])
                        if val > best_val + 1e-12:
                            best_val = val
                            best_a = a
                            improved = True
                    assigned[ag] = best_a
                if not improved:
                    break  # no more local improvement

        # 4) Commit chosen joint action for this step
        # Ensure agents already empty are SILENT
        for ag in range(num_agents):
            if qnow[ag] <= 0.0:
                assigned[ag] = silent_action_idx

        # Log joint action BEFORE all-empty
        joint_actions_pre_empty.append(assigned.copy())

        RRA = rra_from_joint_actions(assigned)
        global_reward, _, _, _ = run_env.step(RRA.copy(), t, 1)
        per_step_rewards_full.append(float(np.array(global_reward).reshape(-1)[0]))

        # If queues have just become all-empty, remaining steps will be handled at loop top

    total_return_full = float(sum(per_step_rewards_full))

    print("joint_actions_pre_empty: ")
    print(joint_actions_pre_empty)
    print("per_step_rewards_full: ")
    print(per_step_rewards_full)
    print("total_return_full: ")
    print(total_return_full)

    return joint_actions_pre_empty, per_step_rewards_full, total_return_full



def plan_SIG_SL_exhaustive(
    params,
    env,
    num_agents: int,
    action_dim: int,
    environ_helper,
    *,
    horizon: int = 50,
    game_mode: int = 2,
    silent_action_idx: Optional[int] = None,
    queue_round_decimals: int = 12
) -> Tuple[List[List[int]], List[float], float]:
    """
    Exhaustive search planner for SIG-SL: at each timestep, brute-force search
    for optimal joint action given current queue states.

    Returns:
      joint_actions_pre_empty : list of joint action-id lists from t=0 until all queues empty
      per_step_rewards_full   : list of length = horizon with env global_reward each step
      total_return_full       : sum(per_step_rewards_full)
    """
    from itertools import product
    
    assert game_mode == 2, "This planner assumes game_mode == 2 reward semantics."

    # ---------- helpers ----------
    def flatten_q(qarr) -> np.ndarray:
        return np.array(qarr, dtype=float).reshape(-1)

    def queues_empty(qarr) -> bool:
        return np.all(flatten_q(qarr) <= 0.0)

    if silent_action_idx is None:
        silent_action_idx = action_dim - 1

    # Map each action-id → (subchannel, power)
    action_to_rra: List[Tuple[int, int]] = []
    discovered_silent = None
    for a in range(action_dim):
        sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[a]]))
        sc, pw = int(sc), int(pw)
        action_to_rra.append((sc, pw))
        if (sc, pw) == (-1, -1):
            discovered_silent = a
    if discovered_silent is not None:
        silent_action_idx = discovered_silent

    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)

    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA

    # ---------- Per-timestep exhaustive search ----------
    joint_actions_pre_empty: List[List[int]] = []
    per_step_rewards_full: List[float] = []

    run_env = copy.deepcopy(env)

    for t in range(horizon):
        qnow = flatten_q(run_env.queue)

        if queues_empty(qnow):
            # After all queues empty: everyone SILENT
            actions = [silent_action_idx] * num_agents
            RRA = rra_from_joint_actions(actions)
            global_reward, _, _, _ = run_env.step(RRA.copy(), t, 1)
            per_step_rewards_full.append(float(np.array(global_reward).reshape(-1)[0]))
            continue

        # 1) Determine valid action space for each agent
        valid_actions_per_agent: List[List[int]] = []
        for ag in range(num_agents):
            if qnow[ag] <= 0.0:
                # Agent has empty queue → must be SILENT
                valid_actions_per_agent.append([silent_action_idx])
            else:
                # Agent has data → can choose any action
                valid_actions_per_agent.append(list(range(action_dim)))

        # 2) Generate all valid joint actions (Cartesian product)
        joint_actions = [list(ja) for ja in product(*valid_actions_per_agent)]

        # 3) Exhaustive search over all valid joint actions
        optimal_joint_action = None
        optimal_reward = -float('inf')

        for ja in joint_actions:
            # Create temporary environment to evaluate this joint action
            tmp_env = copy.deepcopy(run_env)
            RRA = rra_from_joint_actions(ja)
            global_reward, _, _, _ = tmp_env.step(RRA.copy(), t, 1)
            reward_value = float(np.array(global_reward).reshape(-1)[0])

            if reward_value > optimal_reward:
                optimal_reward = reward_value
                optimal_joint_action = ja

        # 4) Execute the optimal joint action
        joint_actions_pre_empty.append(optimal_joint_action.copy())
        RRA = rra_from_joint_actions(optimal_joint_action)
        global_reward, _, _, _ = run_env.step(RRA.copy(), t, 1)
        per_step_rewards_full.append(float(np.array(global_reward).reshape(-1)[0]))

        print(f"t={t}, queues={qnow}, optimal_action={optimal_joint_action}, reward={optimal_reward:.4f}")

    total_return_full = float(sum(per_step_rewards_full))

    print("\n" + "="*60)
    print("EXHAUSTIVE SEARCH RESULTS")
    print("="*60)
    print(f"Total timesteps with decisions: {len(joint_actions_pre_empty)}")
    print(f"Total return: {total_return_full:.4f}")
    print("="*60)

    return joint_actions_pre_empty, per_step_rewards_full, total_return_full


def plan_SIG_SL_max_v2v_throughput(
    params,
    env,
    num_agents: int,
    action_dim: int,
    environ_helper,
    *,
    horizon: int = 50,
    game_mode: int = 2,
    silent_action_idx: Optional[int] = None,
    queue_round_decimals: int = 12
) -> Tuple[List[List[int]], List[float], List[float], float, float]:
    """
    Exhaustive search planner for SIG-SL: at each timestep, brute-force search
    for joint action that MAXIMIZES V2V THROUGHPUT (not global reward).
    
    DETAILED LOGGING VERSION - shows complete episode execution.

    Returns:
      joint_actions_pre_empty : list of joint action-id lists from t=0 until all queues empty
      per_step_v2v_throughput : list of V2V throughput at each timestep
      per_step_global_rewards : list of global rewards at each timestep
      total_v2v_throughput    : sum of V2V throughput
      total_global_reward     : sum of global rewards
    """
    from itertools import product
    import copy
    import numpy as np
    import torch
    
    assert game_mode == 2, "This planner assumes game_mode == 2 reward semantics."

    # ---------- helpers ----------
    def flatten_q(qarr) -> np.ndarray:
        return np.array(qarr, dtype=float).reshape(-1)

    def queues_empty(qarr) -> bool:
        return np.all(flatten_q(qarr) <= 0.0)

    if silent_action_idx is None:
        silent_action_idx = action_dim - 1

    # Map each action-id → (subchannel, power)
    action_to_rra: List[Tuple[int, int]] = []
    discovered_silent = None
    for a in range(action_dim):
        sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[a]]))
        sc, pw = int(sc), int(pw)
        action_to_rra.append((sc, pw))
        if (sc, pw) == (-1, -1):
            discovered_silent = a
    if discovered_silent is not None:
        silent_action_idx = discovered_silent

    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)

    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA

    print("\n" + "="*80)
    print("EXHAUSTIVE SEARCH: MAXIMIZE V2V THROUGHPUT")
    print("="*80)
    print(f"Action space size: {action_dim}")
    print(f"Silent action index: {silent_action_idx}")
    print(f"Silent action maps to: {action_to_rra[silent_action_idx]}")
    print(f"Horizon: {horizon}")
    print("="*80 + "\n")

    # ---------- Per-timestep exhaustive search for MAX V2V THROUGHPUT ----------
    joint_actions_pre_empty: List[List[int]] = []
    per_step_v2v_throughput: List[float] = []
    per_step_global_rewards: List[float] = []

    run_env = copy.deepcopy(env)
    
    all_queues_empty_at = None  # Track when all queues first become empty

    for t in range(horizon):
        qnow = flatten_q(run_env.queue)
        
        print(f"{'='*80}")
        print(f"TIMESTEP t={t}")
        print(f"{'='*80}")
        print(f"Queues: {qnow}")

        if queues_empty(qnow):
            if all_queues_empty_at is None:
                all_queues_empty_at = t
                print(f">>> ALL QUEUES EMPTY at t={t} <<<")
            
            # After all queues empty: everyone SILENT
            actions = [silent_action_idx] * num_agents
            print(f"All agents silent: {actions}")
            
            RRA = rra_from_joint_actions(actions)
            global_reward, individual_rewards, _, _ = run_env.step(RRA.copy(), t, 1)
            v2v_throughput = float(np.sum(individual_rewards))
            global_reward_val = float(np.array(global_reward).reshape(-1)[0])
            
            per_step_v2v_throughput.append(v2v_throughput)
            per_step_global_rewards.append(global_reward_val)
            
            print(f"V2V throughput: {v2v_throughput:.6f}")
            print(f"Global reward:  {global_reward_val:.6f}")
            print(f"(No search - all agents silent)")
            print()
            continue

        # Identify which agents have empty queues
        empty_agents = [ag for ag in range(num_agents) if qnow[ag] <= 0.0]
        active_agents = [ag for ag in range(num_agents) if qnow[ag] > 0.0]
        
        print(f"Empty agents:  {empty_agents}")
        print(f"Active agents: {active_agents}")

        # 1) Determine valid action space for each agent
        valid_actions_per_agent: List[List[int]] = []
        for ag in range(num_agents):
            if qnow[ag] <= 0.0:
                # Agent has empty queue → must be SILENT
                valid_actions_per_agent.append([silent_action_idx])
            else:
                # Agent has data → can choose any action
                valid_actions_per_agent.append(list(range(action_dim)))

        # 2) Generate all valid joint actions (Cartesian product)
        joint_actions = [list(ja) for ja in product(*valid_actions_per_agent)]
        
        print(f"Searching {len(joint_actions)} joint actions...")

        # 3) Exhaustive search over all valid joint actions - MAXIMIZE V2V THROUGHPUT
        optimal_joint_action = None
        max_v2v_throughput = -float('inf')
        corresponding_global_reward = None

        for ja in joint_actions:
            # Create temporary environment to evaluate this joint action
            tmp_env = copy.deepcopy(run_env)
            RRA = rra_from_joint_actions(ja)
            global_reward, individual_rewards, _, _ = tmp_env.step(RRA.copy(), t, 1)
            
            # Calculate V2V throughput (sum of individual_rewards)
            v2v_throughput = float(np.sum(individual_rewards))
            global_reward_val = float(np.array(global_reward).reshape(-1)[0])

            # Maximize V2V throughput, not global reward
            if v2v_throughput > max_v2v_throughput:
                max_v2v_throughput = v2v_throughput
                optimal_joint_action = ja
                corresponding_global_reward = global_reward_val

        print(f"\n>>> OPTIMAL ACTION FOUND <<<")
        print(f"Joint action:   {optimal_joint_action}")
        
        # Show action breakdown
        for ag in range(num_agents):
            sc, pw = action_to_rra[optimal_joint_action[ag]]
            print(f"  Agent {ag}: action={optimal_joint_action[ag]:2d} → Ch={sc:2d}, Power={pw:2d}")
        
        print(f"V2V throughput: {max_v2v_throughput:.6f}")
        print(f"Global reward:  {corresponding_global_reward:.6f}")

        # 4) Execute the optimal joint action (max V2V throughput)
        joint_actions_pre_empty.append(optimal_joint_action.copy())
        RRA = rra_from_joint_actions(optimal_joint_action)
        global_reward, individual_rewards, _, _ = run_env.step(RRA.copy(), t, 1)
        
        v2v_throughput = float(np.sum(individual_rewards))
        global_reward_val = float(np.array(global_reward).reshape(-1)[0])
        
        per_step_v2v_throughput.append(v2v_throughput)
        per_step_global_rewards.append(global_reward_val)
        
        # Show individual agent rewards
        print(f"\nIndividual V2V rewards:")
        for ag in range(num_agents):
            print(f"  Agent {ag}: {individual_rewards[ag][0]:.6f}")
        
        print(f"\nQueues after execution: {flatten_q(run_env.queue)}")
        print()

    total_v2v_throughput = float(sum(per_step_v2v_throughput))
    total_global_reward = float(sum(per_step_global_rewards))

    print("\n" + "="*80)
    print("EXHAUSTIVE SEARCH COMPLETE")
    print("="*80)
    print(f"Total timesteps simulated:      {horizon}")
    print(f"Timesteps with decisions:       {len(joint_actions_pre_empty)}")
    print(f"All queues empty at:            t={all_queues_empty_at}")
    print(f"Timesteps after queues empty:   {horizon - all_queues_empty_at if all_queues_empty_at else 0}")
    print(f"\nTotal V2V throughput:           {total_v2v_throughput:.6f}")
    print(f"Total global reward:            {total_global_reward:.6f}")
    
    # Breakdown by phase
    if all_queues_empty_at is not None and all_queues_empty_at < horizon:
        v2v_before_empty = sum(per_step_v2v_throughput[:all_queues_empty_at])
        v2v_after_empty = sum(per_step_v2v_throughput[all_queues_empty_at:])
        global_before_empty = sum(per_step_global_rewards[:all_queues_empty_at])
        global_after_empty = sum(per_step_global_rewards[all_queues_empty_at:])
        
        print(f"\nBREAKDOWN:")
        print(f"  Before all empty (t=0 to t={all_queues_empty_at-1}):")
        print(f"    V2V throughput:  {v2v_before_empty:.6f}")
        print(f"    Global reward:   {global_before_empty:.6f}")
        print(f"  After all empty (t={all_queues_empty_at} to t={horizon-1}):")
        print(f"    V2V throughput:  {v2v_after_empty:.6f}")
        print(f"    Global reward:   {global_after_empty:.6f}")
        print(f"    Avg per step:    {global_after_empty/(horizon-all_queues_empty_at):.6f}")
    
    print("="*80 + "\n")

    return joint_actions_pre_empty, per_step_v2v_throughput, per_step_global_rewards, total_v2v_throughput, total_global_reward


import time
import numpy as np

class DetailedTrainingTracker:
    """Detailed training tracker to compare IDQL vs QMIX performance"""
    
    def __init__(self, algorithm_name, num_agents):
        self.algorithm_name = algorithm_name
        self.num_agents = num_agents
        self.reset_episode()
        self.all_episodes = []
        
    def reset_episode(self):
        """Reset for new episode"""
        self.episode_start = time.time()
        self.training_calls = []
        self.timestep_count = 0
        self.total_training_time = 0
        self.env_step_time = 0
        self.env_step_start = None
        
    def start_env_step(self):
        """Mark start of environment interaction"""
        self.env_step_start = time.time()
        
    def end_env_step(self):
        """Mark end of environment interaction"""
        if self.env_step_start:
            self.env_step_time += time.time() - self.env_step_start
            self.timestep_count += 1
            
    def record_training_call(self, duration, batch_size=1, seq_len=1):
        """Record a single training call"""
        self.training_calls.append({
            'duration': duration,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'transitions': batch_size * seq_len
        })
        self.total_training_time += duration
        
    def end_episode(self, episode_num):
        """Complete episode and calculate statistics"""
        episode_duration = time.time() - self.episode_start
        
        # Calculate statistics
        num_training_calls = len(self.training_calls)
        if num_training_calls > 0:
            avg_call_duration = np.mean([c['duration'] for c in self.training_calls])
            total_transitions = sum([c['transitions'] for c in self.training_calls])
            transitions_per_second = total_transitions / self.total_training_time if self.total_training_time > 0 else 0
        else:
            avg_call_duration = 0
            total_transitions = 0
            transitions_per_second = 0
            
        # Store episode data
        episode_data = {
            'episode_num': episode_num,
            'total_duration': episode_duration,
            'training_time': self.total_training_time,
            'env_time': self.env_step_time,
            'other_time': episode_duration - self.total_training_time - self.env_step_time,
            'num_training_calls': num_training_calls,
            'timesteps': self.timestep_count,
            'avg_call_duration': avg_call_duration,
            'total_transitions': total_transitions,
            'transitions_per_second': transitions_per_second,
            'training_percentage': (self.total_training_time / episode_duration * 100) if episode_duration > 0 else 0
        }
        
        self.all_episodes.append(episode_data)
        
        # Print episode summary
        print(f"\n{'='*70}")
        print(f"{self.algorithm_name} - Episode {episode_num} Timing Summary")
        print(f"{'='*70}")
        print(f"Total episode time:      {episode_duration:.2f}s")
        print(f"  - Training time:       {self.total_training_time:.2f}s ({episode_data['training_percentage']:.1f}%)")
        print(f"  - Environment time:    {self.env_step_time:.2f}s ({self.env_step_time/episode_duration*100:.1f}%)")
        print(f"  - Other overhead:      {episode_data['other_time']:.2f}s ({episode_data['other_time']/episode_duration*100:.1f}%)")
        print(f"Training calls:          {num_training_calls} (over {self.timestep_count} timesteps)")
        print(f"Avg duration per call:   {avg_call_duration:.4f}s")
        print(f"Total transitions:       {total_transitions}")
        print(f"Transitions/second:      {transitions_per_second:.1f}")
        print(f"{'='*70}\n")
        
        self.reset_episode()
        return episode_data
        
    def print_comparison_summary(self, last_n_episodes=5):
        """Print summary comparing last N episodes"""
        if len(self.all_episodes) < last_n_episodes:
            return
            
        recent = self.all_episodes[-last_n_episodes:]
        
        print(f"\n{'#'*70}")
        print(f"{self.algorithm_name} - Summary of Last {last_n_episodes} Episodes")
        print(f"{'#'*70}")
        print(f"Average episode duration:     {np.mean([e['total_duration'] for e in recent]):.2f}s")
        print(f"Average training time:        {np.mean([e['training_time'] for e in recent]):.2f}s")
        print(f"Average training percentage:  {np.mean([e['training_percentage'] for e in recent]):.1f}%")
        print(f"Average training calls:       {np.mean([e['num_training_calls'] for e in recent]):.1f}")
        print(f"Average call duration:        {np.mean([e['avg_call_duration'] for e in recent]):.4f}s")
        print(f"Average transitions/second:   {np.mean([e['transitions_per_second'] for e in recent]):.1f}")
        print(f"{'#'*70}\n")

def compare_action_vs_max_v2v(
    params,
    env,
    joint_action: list,
    environ_helper,
    current_timestep: int = 0
):
    """
    Compare a given joint action against max V2V throughput with same silent agents.
    CORRECTED: Now displays V2I throughput and its contribution to global reward.
    
    Args:
        params: Environment parameters
        env: Current environment state (will be deep copied)
        joint_action: List of action indices, e.g., [12, 3, 6, 12]
        environ_helper: Helper for action→RRA mapping
        current_timestep: Current timestep for env.step()
        
    Prints:
        - Which agents are silent
        - Your action's V2V throughput, V2I throughput, and global reward
        - Max V2V action and its throughput/V2I/reward
        - The gap between them
        
    Returns:
        Dictionary with comparison results
    """
    import copy
    import numpy as np
    import torch
    from itertools import product
    
    num_agents = len(joint_action)
    
    # Discover action space size by trying actions until we find silent action
    action_dim = 0
    silent_action_idx = None
    action_to_rra = []
    
    for a in range(100):  # Try up to 100 actions
        try:
            sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[a]]))
            sc, pw = int(sc), int(pw)
            action_to_rra.append((sc, pw))
            if (sc, pw) == (-1, -1):
                silent_action_idx = a
                action_dim = a + 1
                break
            action_dim = a + 1
        except:
            break
    
    if silent_action_idx is None:
        silent_action_idx = action_dim - 1
    
    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)
    
    def rra_from_joint_actions(joint_actions: list) -> np.ndarray:
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA
    
    # Step 1: Identify silent agents in your action
    silent_agents = []
    for ag in range(num_agents):
        if action_to_rra[joint_action[ag]] == (-1, -1):
            silent_agents.append(ag)
    
    print("\n" + "="*80)
    print("COMPARISON: Your Action vs Max V2V Throughput")
    print("="*80)
    print(f"Your joint action:     {joint_action}")
    print(f"Silent agents:         {silent_agents}")
    print(f"Active agents:         {num_agents - len(silent_agents)}")
    print(f"V2V weight:            {env.V2V_weight}")
    print(f"V2I weight:            {env.V2I_weight}")
    print("-"*80)
    
    # Step 2: Execute your action
    your_env = copy.deepcopy(env)
    RRA_yours = rra_from_joint_actions(joint_action)
    your_global_reward, your_individual_rewards, your_V2I_SE, _ = your_env.step(RRA_yours.copy(), current_timestep, 1)
    
    # Extract values
    # individual_rewards already contains weighted V2V: V2V_SE * 0.01 * V2V_weight
    your_v2v_weighted = float(np.sum(your_individual_rewards))
    your_v2i_raw = float(np.sum(your_V2I_SE))  # Raw V2I throughput in Mbps
    your_v2i_weighted = your_v2i_raw * 0.01 * env.V2I_weight  # V2I contribution to global reward
    your_global = float(np.array(your_global_reward).reshape(-1)[0])
    
    print(f"\nYOUR ACTION:")
    print(f"  V2V (weighted):       {your_v2v_weighted:.6f}")
    print(f"  V2I throughput:       {your_v2i_raw:.6f} Mbps")
    print(f"  V2I (weighted):       {your_v2i_weighted:.6f}")
    print(f"  Global reward:        {your_global:.6f}")
    print(f"    Verify: {your_v2v_weighted:.6f} + {your_v2i_weighted:.6f} = {your_v2v_weighted + your_v2i_weighted:.6f}")
    
    # Step 3: Find max V2V throughput with same silent agents
    valid_actions_per_agent = []
    for ag in range(num_agents):
        if ag in silent_agents:
            valid_actions_per_agent.append([silent_action_idx])
        else:
            valid_actions_per_agent.append(list(range(action_dim)))
    
    joint_actions_space = [list(ja) for ja in product(*valid_actions_per_agent)]
    
    print(f"\nSearching through {len(joint_actions_space)} joint actions for max V2V...")
    
    optimal_joint_action = None
    max_v2v_weighted = -float('inf')
    optimal_v2i_raw = None
    optimal_v2i_weighted = None
    optimal_global_reward = None
    
    for ja in joint_actions_space:
        tmp_env = copy.deepcopy(env)
        RRA = rra_from_joint_actions(ja)
        global_reward, individual_rewards, V2I_SE, _ = tmp_env.step(RRA.copy(), current_timestep, 1)
        
        v2v_weighted = float(np.sum(individual_rewards))
        
        if v2v_weighted > max_v2v_weighted:
            max_v2v_weighted = v2v_weighted
            optimal_joint_action = ja
            optimal_v2i_raw = float(np.sum(V2I_SE))
            optimal_v2i_weighted = optimal_v2i_raw * 0.01 * env.V2I_weight
            optimal_global_reward = float(np.array(global_reward).reshape(-1)[0])
    
    print(f"\nMAX V2V THROUGHPUT ACTION:")
    print(f"  Joint action:         {optimal_joint_action}")
    print(f"  V2V (weighted):       {max_v2v_weighted:.6f}")
    print(f"  V2I throughput:       {optimal_v2i_raw:.6f} Mbps")
    print(f"  V2I (weighted):       {optimal_v2i_weighted:.6f}")
    print(f"  Global reward:        {optimal_global_reward:.6f}")
    print(f"    Verify: {max_v2v_weighted:.6f} + {optimal_v2i_weighted:.6f} = {max_v2v_weighted + optimal_v2i_weighted:.6f}")
    
    print(f"\nCOMPARISON:")
    v2v_gap = your_v2v_weighted - max_v2v_weighted
    v2v_gap_pct = (v2v_gap / max_v2v_weighted * 100) if max_v2v_weighted != 0 else 0
    
    v2i_gap = your_v2i_weighted - optimal_v2i_weighted
    v2i_gap_pct = (v2i_gap / optimal_v2i_weighted * 100) if optimal_v2i_weighted != 0 else 0
    
    global_gap = your_global - optimal_global_reward
    global_gap_pct = (global_gap / optimal_global_reward * 100) if optimal_global_reward != 0 else 0
    
    print(f"  V2V gap:              {v2v_gap:+.6f} ({v2v_gap_pct:+.2f}%)")
    print(f"  V2I gap:              {v2i_gap:+.6f} ({v2i_gap_pct:+.2f}%)")
    print(f"  Global reward gap:    {global_gap:+.6f} ({global_gap_pct:+.2f}%)")
    
    if abs(v2v_gap) < 0.0001:
        print(f"\n  → Your action MATCHES max V2V throughput ✓")
    elif v2v_gap < 0:
        print(f"\n  → Your action has LOWER V2V throughput")
        if global_gap > 0:
            print(f"     BUT higher global reward (better V2I: {v2i_gap:+.6f})")
    else:
        print(f"\n  → Your action EXCEEDS max V2V throughput (impossible - check code)")
    
    print("="*80 + "\n")
    
    return {
        'your_action': joint_action,
        'your_v2v_weighted': your_v2v_weighted,
        'your_v2i_raw': your_v2i_raw,
        'your_v2i_weighted': your_v2i_weighted,
        'your_global': your_global,
        'max_v2v_action': optimal_joint_action,
        'max_v2v_weighted': max_v2v_weighted,
        'max_v2i_raw': optimal_v2i_raw,
        'max_v2i_weighted': optimal_v2i_weighted,
        'max_global': optimal_global_reward,
        'v2v_gap': v2v_gap,
        'v2i_gap': v2i_gap,
        'global_gap': global_gap,
        'silent_agents': silent_agents
    }


def compare_action_vs_max_global(
    params,
    env,
    joint_action: list,
    environ_helper,
    current_timestep: int = 0
):
    """
    Compare a given joint action against max GLOBAL REWARD with same silent agents.
    
    This searches for the action that maximizes global reward (weighted V2V + weighted V2I).
    
    Args:
        params: Environment parameters
        env: Current environment state (will be deep copied)
        joint_action: List of action indices, e.g., [12, 3, 6, 12]
        environ_helper: Helper for action→RRA mapping
        current_timestep: Current timestep for env.step()
        
    Returns:
        Dictionary with comparison results
    """
    import copy
    import numpy as np
    import torch
    from itertools import product
    
    num_agents = len(joint_action)
    
    # Discover action space size
    action_dim = 0
    silent_action_idx = None
    action_to_rra = []
    
    for a in range(100):
        try:
            sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[a]]))
            sc, pw = int(sc), int(pw)
            action_to_rra.append((sc, pw))
            if (sc, pw) == (-1, -1):
                silent_action_idx = a
                action_dim = a + 1
                break
            action_dim = a + 1
        except:
            break
    
    if silent_action_idx is None:
        silent_action_idx = action_dim - 1
    
    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)
    
    def rra_from_joint_actions(joint_actions: list) -> np.ndarray:
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA
    
    # Identify silent agents
    silent_agents = []
    for ag in range(num_agents):
        if action_to_rra[joint_action[ag]] == (-1, -1):
            silent_agents.append(ag)
    
    print("\n" + "="*80)
    print("COMPARISON: Your Action vs Max GLOBAL REWARD")
    print("="*80)
    print(f"Your joint action:     {joint_action}")
    print(f"Silent agents:         {silent_agents}")
    print(f"Active agents:         {num_agents - len(silent_agents)}")
    print(f"V2V weight:            {env.V2V_weight}")
    print(f"V2I weight:            {env.V2I_weight}")
    print("-"*80)
    
    # Execute your action
    your_env = copy.deepcopy(env)
    RRA_yours = rra_from_joint_actions(joint_action)
    your_global_reward, your_individual_rewards, your_V2I_SE, _ = your_env.step(RRA_yours.copy(), current_timestep, 1)
    
    your_v2v_weighted = float(np.sum(your_individual_rewards))
    your_v2i_raw = float(np.sum(your_V2I_SE))
    your_v2i_weighted = your_v2i_raw * 0.01 * env.V2I_weight
    your_global = float(np.array(your_global_reward).reshape(-1)[0])
    
    print(f"\nYOUR ACTION:")
    print(f"  V2V (weighted):       {your_v2v_weighted:.6f}")
    print(f"  V2I throughput:       {your_v2i_raw:.6f} Mbps")
    print(f"  V2I (weighted):       {your_v2i_weighted:.6f}")
    print(f"  Global reward:        {your_global:.6f}")
    
    # Find max global reward
    valid_actions_per_agent = []
    for ag in range(num_agents):
        if ag in silent_agents:
            valid_actions_per_agent.append([silent_action_idx])
        else:
            valid_actions_per_agent.append(list(range(action_dim)))
    
    joint_actions_space = [list(ja) for ja in product(*valid_actions_per_agent)]
    
    print(f"\nSearching through {len(joint_actions_space)} joint actions for max global reward...")
    
    optimal_joint_action = None
    max_global_reward = -float('inf')
    optimal_v2v_weighted = None
    optimal_v2i_raw = None
    optimal_v2i_weighted = None
    
    for ja in joint_actions_space:
        tmp_env = copy.deepcopy(env)
        RRA = rra_from_joint_actions(ja)
        global_reward, individual_rewards, V2I_SE, _ = tmp_env.step(RRA.copy(), current_timestep, 1)
        
        global_reward_val = float(np.array(global_reward).reshape(-1)[0])
        
        if global_reward_val > max_global_reward:
            max_global_reward = global_reward_val
            optimal_joint_action = ja
            optimal_v2v_weighted = float(np.sum(individual_rewards))
            optimal_v2i_raw = float(np.sum(V2I_SE))
            optimal_v2i_weighted = optimal_v2i_raw * env.V2I_weight
    
    print(f"\nMAX GLOBAL REWARD ACTION:")
    print(f"  Joint action:         {optimal_joint_action}")
    print(f"  V2V (weighted):       {optimal_v2v_weighted:.6f}")
    print(f"  V2I throughput:       {optimal_v2i_raw:.6f} Mbps")
    print(f"  V2I (weighted):       {optimal_v2i_weighted:.6f}")
    print(f"  Global reward:        {max_global_reward:.6f}")
    
    print(f"\nCOMPARISON:")
    v2v_gap = your_v2v_weighted - optimal_v2v_weighted
    v2v_gap_pct = (v2v_gap / optimal_v2v_weighted * 100) if optimal_v2v_weighted != 0 else 0
    
    v2i_gap = your_v2i_weighted - optimal_v2i_weighted
    v2i_gap_pct = (v2i_gap / optimal_v2i_weighted * 100) if optimal_v2i_weighted != 0 else 0
    
    global_gap = your_global - max_global_reward
    global_gap_pct = (global_gap / max_global_reward * 100) if max_global_reward != 0 else 0
    
    print(f"  V2V gap:              {v2v_gap:+.6f} ({v2v_gap_pct:+.2f}%)")
    print(f"  V2I gap:              {v2i_gap:+.6f} ({v2i_gap_pct:+.2f}%)")
    print(f"  Global reward gap:    {global_gap:+.6f} ({global_gap_pct:+.2f}%)")
    
    if abs(global_gap) < 0.0001:
        print(f"\n  → Your action is OPTIMAL ✓✓✓")
    elif global_gap < 0:
        print(f"\n  → Your action is SUBOPTIMAL")
        if abs(v2v_gap) < 0.0001:
            print(f"     (V2V is optimal, but V2I can be improved)")
        elif abs(v2i_gap) < 0.0001:
            print(f"     (V2I is optimal, but V2V can be improved)")
    else:
        print(f"\n  → Your action EXCEEDS optimal (impossible - check code)")
    
    print("="*80 + "\n")
    
    return {
        'your_action': joint_action,
        'your_v2v_weighted': your_v2v_weighted,
        'your_v2i_raw': your_v2i_raw,
        'your_v2i_weighted': your_v2i_weighted,
        'your_global': your_global,
        'optimal_action': optimal_joint_action,
        'optimal_v2v_weighted': optimal_v2v_weighted,
        'optimal_v2i_raw': optimal_v2i_raw,
        'optimal_v2i_weighted': optimal_v2i_weighted,
        'optimal_global': max_global_reward,
        'v2v_gap': v2v_gap,
        'v2i_gap': v2i_gap,
        'global_gap': global_gap,
        'silent_agents': silent_agents
    }


# Example usage:
"""
# Compare against max V2V throughput (with same silent agents)
result = compare_action_vs_max_v2v(
    params=params,
    env=env,
    joint_action=[12, 3, 6, 12],
    environ_helper=environ_helper,
    current_timestep=0
)

# Compare against max global reward (weighted V2V + weighted V2I)
result = compare_action_vs_max_global(
    params=params,
    env=env,
    joint_action=[12, 3, 6, 12],
    environ_helper=environ_helper,
    current_timestep=0
)
"""


def plan_random_baseline(
    params,
    env,
    num_agents: int,
    action_dim: int,
    environ_helper,
    *,
    horizon: int = 50,
    game_mode: int = 2,
    num_trials: int = 10  # Average over multiple random trials for stability
) -> Tuple[List[List[int]], List[float], float]:
    """
    Random action baseline for comparison.
    
    Returns:
      joint_actions_all : list of joint action-id lists for each timestep
      per_step_rewards  : list of length = horizon with env global_reward each step
      total_return      : sum(per_step_rewards)
    """
    
    n_neighbor = int(getattr(params, "n_neighbor", 1) or 1)
    
    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        """Convert joint action IDs to RRA format."""
        RRA = np.zeros((num_agents, n_neighbor, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = environ_helper.mapping_action2RRA(torch.tensor([[joint_actions[ag]]]))
            RRA[ag, 0, 0] = int(sc)
            RRA[ag, 0, 1] = int(pw)
        return RRA
    
    # Average over multiple trials to reduce variance
    all_trial_returns = []
    
    for trial in range(num_trials):
        # IMPORTANT: Reset environment for each trial
        env.new_random_game()  # ← ADD THIS LINE
        
        joint_actions_all: List[List[int]] = []
        per_step_rewards: List[float] = []
        
        for t in range(horizon):
            # Generate random joint action
            random_joint_action = [np.random.randint(0, action_dim) for _ in range(num_agents)]
            joint_actions_all.append(random_joint_action)
            
            # Convert to RRA and execute
            RRA = rra_from_joint_actions(random_joint_action)
            global_reward, _, _, _ = env.step(RRA.copy(), t, 1)
            per_step_rewards.append(float(np.array(global_reward).reshape(-1)[0]))
        
        trial_return = float(sum(per_step_rewards))
        all_trial_returns.append(trial_return)
    
    # Return the average over all trials
    avg_total_return = float(np.mean(all_trial_returns))
    
    print(f"Random baseline returns over {num_trials} trials: {all_trial_returns}")
    print(f"Average random baseline return: {avg_total_return:.3f}")
    
    # For consistency with greedy planner API, return last trial's actions/rewards
    # but the averaged total_return
    return joint_actions_all, per_step_rewards, avg_total_return