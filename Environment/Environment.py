"""
V2X Resource Allocation Environment for Multi-Agent Reinforcement Learning.

Original work by Dr. Liu Tong
Revised by Mei Jie, 2023-11-22
Refactored 2026 Jan 19th
Refactored 2026 Jan 24th - Removed game_mode, use task_type directly
"""

from __future__ import division
import numpy as np
import math
import sys


# =============================================================================
# Vehicle Entity
# =============================================================================

class Vehicle:
    """Vehicle entity with position, velocity, and communication destinations."""

    def __init__(self, start_position, velocity):
        self.position = start_position
        self.velocity = velocity
        self.destinations = []


# =============================================================================
# V2X Environment
# =============================================================================

class Environ:
    """V2X Communication Environment for Multi-Agent Reinforcement Learning."""

    def __init__(self, params):
        # ---------------------------------------------------------------------
        # State variables (initialized to None, set during episode)
        # ---------------------------------------------------------------------
        self.individual_rewards = None
        self.total_v2v_interference_dbm = None
        self.v2v_pathloss = None
        self.v2v_pathloss_with_ff = None
        self.active_links = None
        self.queue = None
        self.vehicles_v2v = []
        self.v2v_spectral_efficiency = None
        self.total_v2v_interference = None
        self.v2v_interference_at_rx = None
        self.v2v_interference_to_others = None
        self.previous_interference_per_sc = None

        # V2I state variables
        self.v2i_pathloss_to_veh = None
        self.v2i_pathloss_to_veh_with_ff = None
        self.v2i_signals = None
        self.v2i_spectral_efficiency = None
        self.vehicles_v2i = []
        self.v2v_interference_to_v2i = None
        self.v2v_pathloss_to_bs = None
        self.v2v_pathloss_to_bs_with_ff = None
        self.v2i_pathloss_to_bs = None
        self.v2i_pathloss_to_bs_with_ff = None

        # ---------------------------------------------------------------------
        # Channel model parameters
        # ---------------------------------------------------------------------
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.bs_antenna_height = 25
        self.carrier_freq_ghz = 2
        self.decorrelation_distance_m = params.decorrelation_distance_m
        self.shadow_std_db = params.shadow_std_db

        # ---------------------------------------------------------------------
        # Reward configuration
        # ---------------------------------------------------------------------
        self.reward_queue_empty = params.reward_queue_empty
        self.v2v_weight = params.v2v_weight
        self.v2i_weight = params.v2i_weight
        self.reward_lambda1 = params.reward_lambda1
        self.reward_lambda2 = params.reward_lambda2
        self.reward_lambda3 = params.reward_lambda3

        # ---------------------------------------------------------------------
        # V2I settings
        # ---------------------------------------------------------------------
        self.v2i_power_dbm = params.v2i_power_dbm
        self.bs_position = [0, -35]
        self.distance_between_v2i_veh = 33

        # ---------------------------------------------------------------------
        # Network topology
        # ---------------------------------------------------------------------
        self.n_sc = params.n_sc
        self.n_veh = params.n_veh
        self.n_veh_per_platoon = params.n_veh_per_platoon
        self.n_agent = params.n_agent
        self.agent_to_veh = params.agent_to_veh
        self.train_data = params.train_data
        self.distance_between_veh = 16.5
        self.n_neighbor = params.n_neighbor
        self.v2v_power_levels_dbm = params.v2v_power_levels_dbm
        self.noise_power_dbm = params.noise_power_dbm
        self.noise_power_mw = 10 ** (self.noise_power_dbm / 10)
        self.norm_v2v_channel_factor = params.norm_v2v_channel_factor
        self.norm_v2v_interference_factor = params.norm_v2v_interference_factor
        self.bs_antenna_gain_dbi = params.bs_antenna_gain_dbi
        self.bs_noise_figure_db = params.bs_noise_figure_db
        self.veh_antenna_gain_dbi = params.veh_antenna_gain_dbi
        self.veh_noise_figure_db = params.veh_noise_figure_db
        self.fast_fading_enabled = params.fast_fading_enabled

        # ---------------------------------------------------------------------
        # Timing and communication
        # ---------------------------------------------------------------------
        self.time_fast = 0.001
        self.time_slow = params.n_step_per_episode / 1000
        self.n_step_per_episode = int(params.n_step_per_episode)
        self.bandwidth_hz = int(params.bandwidth_per_sc_hz)
        self.max_queue_length = params.max_queue_length
        self.cam_size_bits = params.cam_size_bits
        # self.t_max = params.t_max

        # ---------------------------------------------------------------------
        # Normalization bounds
        # ---------------------------------------------------------------------
        self.static_norm = False
        self.pathloss_bounds = {
            "v2v_link": (44.5, 119.3),
            "v2v_interference": (44.5, 125.6),
            "veh_to_bs": (76.9, 128.1),
        }
        self.inr_norm_bounds = (20.0, 85.0)

        # ---------------------------------------------------------------------
        # Action space configuration
        # ---------------------------------------------------------------------
        self.n_power_levels = params.n_power_levels
        self.n_actions = params.n_actions
        self.state_type = params.state_type

        # ---------------------------------------------------------------------
        # State encoding configuration
        # ---------------------------------------------------------------------
        self.timestep_encoding_type = 'one-hot'

        # ---------------------------------------------------------------------
        # Task type and state dimensions
        # ---------------------------------------------------------------------
        self.task_type = params.task_type

        if self.timestep_encoding_type == 'normalized':
            timestep_dims = 1
        else:
            timestep_dims = self.n_step_per_episode

        if self.task_type == "NFIG":
            # State: G_i + G_ji
            self.state_dim = (self.n_agent) + (self.n_agent) * (self.n_agent - 1)
        elif self.task_type == "SIG":
            # State: t + G_i + G_ji + G_m + G_Bi + G_iB + I_prev + queue
            self.state_dim = ((timestep_dims) + (self.n_agent) + (self.n_agent) * (self.n_agent - 1) +
                              (self.n_sc) + (self.n_sc * self.n_agent) + (self.n_agent) +
                              (self.n_agent * self.n_sc) + (self.n_agent))
        elif self.task_type == "POSIG":
            # Observation: t + G_i + G_iB + I_prev + queue (per agent)
            self.state_dim = (timestep_dims) + (1) + (self.n_agent - 1) + (1)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

        self.local_state_dim = (timestep_dims) + (1) + (1) + (self.n_sc) + (1)

    # =========================================================================
    # Action Mapping
    # =========================================================================

    def map_action_to_rra(self, action, agent_idx=None):
        """
        Map discrete action index to Radio Resource Allocation (SC index, power level).

        Args:
            action: Discrete action index
            agent_idx: Agent index (required for simplified_version state type)

        Returns:
            tuple: (subchannel_index, power_level_index)
                   Returns (-1, -1) for the "no transmission" action
        """
        if action >= self.n_actions - 1:
            return -1, -1

        if self.state_type == 'simplified_version':
            if agent_idx is None:
                raise ValueError("agent_idx required for simplified_version state type")
            sc_idx = agent_idx
            power_level_idx = action % self.n_power_levels
        else:
            if hasattr(action, 'cpu'):
                action_val = action.cpu().numpy()
            else:
                action_val = action
            sc_idx = int(np.floor(action_val / self.n_power_levels))
            power_level_idx = action % self.n_power_levels

        return sc_idx, power_level_idx

    # =========================================================================
    # Path Loss Models
    # =========================================================================

    def _compute_v2v_pathloss(self, position_a, position_b):
        """Calculate V2V path loss using 3GPP Urban Micro LOS model."""
        d1 = abs(position_a[0] - position_b[0])
        d2 = abs(position_a[1] - position_b[1])
        d = math.hypot(d1, d2)
        d_bp = 4 * (self.h_bs) * (self.h_ms) * self.carrier_freq_ghz * (10 ** 9) / (3 * 10 ** 8)

        if d <= 3:
            return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.carrier_freq_ghz / 5)
        elif d < d_bp:
            return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.carrier_freq_ghz / 5)
        else:
            return (40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) -
                    17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.carrier_freq_ghz / 5))

    def _compute_v2v_pathloss_with_ff(self, position_a, position_b):
        """Calculate V2V path loss with Rayleigh fast fading."""
        pathloss_db = self._compute_v2v_pathloss(position_a, position_b)
        g = np.random.exponential(scale=1.0)
        fading_db = 10.0 * np.log10(g)
        return pathloss_db - fading_db

    def _compute_v2i_pathloss(self, position_a):
        """Calculate V2I path loss using 3GPP Urban Macro model."""
        d1 = abs(position_a[0] - self.bs_position[0])
        d2 = abs(position_a[1] - self.bs_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.bs_antenna_height - self.h_ms) ** 2) / 1000)

    def _compute_v2i_pathloss_with_ff(self, position_a):
        """Calculate V2I path loss with Rayleigh fast fading."""
        pathloss_db = self._compute_v2i_pathloss(position_a)
        g = np.random.exponential(scale=1.0)
        fading_db = 10.0 * np.log10(g)
        return pathloss_db - fading_db

    # =========================================================================
    # Fast Fading
    # =========================================================================

    def _renew_fast_fading(self):
        """Update fast fading realizations for all channels."""
        self.v2v_pathloss_with_ff = self.v2v_pathloss.copy()
        for i in range(self.n_veh):
            for j in range(i + 1, self.n_veh):
                pl_ij = self._compute_v2v_pathloss_with_ff(
                    self.vehicles_v2v[i].position, self.vehicles_v2v[j].position)
                self.v2v_pathloss_with_ff[i, j] = pl_ij
                self.v2v_pathloss_with_ff[j, i] = pl_ij

        self.v2i_pathloss_to_bs_with_ff = self.v2i_pathloss_to_bs.copy()
        for sc in range(self.n_sc):
            self.v2i_pathloss_to_bs_with_ff[sc] = self._compute_v2i_pathloss_with_ff(
                self.vehicles_v2i[sc].position)

        self.v2i_pathloss_to_veh_with_ff = self.v2i_pathloss_to_veh.copy()
        for sc in range(self.n_sc):
            for veh_idx in range(self.n_veh):
                pl_sc_veh = self._compute_v2v_pathloss_with_ff(
                    self.vehicles_v2i[sc].position, self.vehicles_v2v[veh_idx].position)
                self.v2i_pathloss_to_veh_with_ff[sc, veh_idx] = pl_sc_veh

        self.v2v_pathloss_to_bs_with_ff = self.v2v_pathloss_to_bs.copy()
        for veh_idx in range(self.n_veh):
            pl_veh_bs = self._compute_v2i_pathloss_with_ff(
                self.vehicles_v2v[veh_idx].position)
            self.v2v_pathloss_to_bs_with_ff[veh_idx] = pl_veh_bs

    # =========================================================================
    # Vehicle Management
    # =========================================================================

    def _add_v2v_vehicle(self, start_position, start_velocity):
        """Add a V2V vehicle to the environment."""
        self.vehicles_v2v.append(Vehicle(start_position, start_velocity))

    def _add_v2i_vehicle(self, start_position, start_velocity):
        """Add a V2I vehicle to the environment."""
        self.vehicles_v2i.append(Vehicle(start_position, start_velocity))

    def _add_vehicles_from_data(self, veh_pos_data):
        """
        Add vehicles using a SINGLE snapshot dataframe (already sampled by caller).
        Assumes veh_pos_data only contains ONE time snapshot.
        """

        # -----------------------------
        # Add V2V vehicles
        # -----------------------------
        veh_ids = []
        for i, num_vehicles in enumerate(self.n_veh_per_platoon):
            for j in range(num_vehicles):
                veh_ids.append(f"carflow{i}_{j}.0")

        df_v2v = veh_pos_data[veh_pos_data['id'].isin(veh_ids)]

        positions = df_v2v[['x', 'y']].values.tolist()
        velocities = df_v2v[['speed']].values.tolist()

        n_v2v = sum(self.n_veh_per_platoon)

        for i in range(n_v2v):
            self._add_v2v_vehicle(positions[i], velocities[i][0])

        # -----------------------------
        # Add V2I vehicles
        # -----------------------------
        veh_ids = [f"carflowV2I_{j}.0" for j in range(self.n_sc)]
        df_v2i = veh_pos_data[veh_pos_data['id'].isin(veh_ids)]

        positions = df_v2i[['x', 'y']].values.tolist()
        velocities = df_v2i[['speed']].values.tolist()

        for i in range(self.n_sc):
            self._add_v2i_vehicle(positions[i], velocities[i][0])



    def _update_positions_from_data(self, interval_idx: int):
        """Update vehicle positions from SUMO data for given time interval."""
        df = self.train_data
        unique_times = np.sort(df["time"].unique())
        if interval_idx - 1 >= len(unique_times):
            raise IndexError("interval_idx exceeds available time-steps")

        t_val = unique_times[interval_idx - 1]
        df_t = df[df["time"] == t_val]

        v2v_ids = [f"carflow{i}_{j}.0"
                   for i, n in enumerate(self.n_veh_per_platoon)
                   for j in range(n)]
        v2v_rows = df_t[df_t["id"].isin(v2v_ids)]

        for k, row in enumerate(v2v_rows.itertuples()):
            if k >= len(self.vehicles_v2v):
                break
            self.vehicles_v2v[k].position = [row.x, row.y]
            self.vehicles_v2v[k].velocity = row.speed

        v2i_ids = [f"carflowV2I_{j}.0" for j in range(self.n_sc)]
        v2i_rows = df_t[df_t["id"].isin(v2i_ids)]

        for k, row in enumerate(v2i_rows.itertuples()):
            if k >= len(self.vehicles_v2i):
                break
            self.vehicles_v2i[k].position = [row.x, row.y]
            self.vehicles_v2i[k].velocity = row.speed

    def _renew_neighbors(self):
        """Set communication destinations for each V2V agent."""
        for i in range(self.n_agent):
            veh = self.agent_to_veh[i]
            self.vehicles_v2v[veh].actions = []

        for i in range(self.n_agent):
            veh = self.agent_to_veh[i]
            destination = [veh + 1]
            self.vehicles_v2v[veh].destinations = destination

    def _renew_channels(self):
        """Update slow fading channel gains for all links."""
        self.v2i_pathloss_to_veh = np.zeros((self.n_sc, self.n_veh))
        for sc_idx in range(self.n_sc):
            for veh_idx in range(self.n_veh):
                self.v2i_pathloss_to_veh[sc_idx, veh_idx] = self._compute_v2v_pathloss(
                    self.vehicles_v2v[veh_idx].position, self.vehicles_v2i[sc_idx].position)

        self.v2i_pathloss_to_bs = np.zeros(self.n_sc)
        for sc_idx in range(self.n_sc):
            self.v2i_pathloss_to_bs[sc_idx] = self._compute_v2i_pathloss(
                self.vehicles_v2i[sc_idx].position)

        self.v2v_pathloss_to_bs = np.zeros(self.n_veh)
        for veh_idx in range(self.n_veh):
            self.v2v_pathloss_to_bs[veh_idx] = self._compute_v2i_pathloss(
                self.vehicles_v2v[veh_idx].position)

        self.v2v_pathloss = np.zeros((self.n_veh, self.n_veh))
        for i in range(self.n_veh):
            for j in range(i + 1, self.n_veh):
                pl = self._compute_v2v_pathloss(
                    self.vehicles_v2v[i].position, self.vehicles_v2v[j].position)
                self.v2v_pathloss[i, j] = pl
                self.v2v_pathloss[j, i] = pl

    # =========================================================================
    # Reward and Performance Computation
    # =========================================================================

    def _compute_performance_and_reward(self, actions_power):
        """Compute spectral efficiency and rewards for all links."""
        actions = actions_power[:, :, 0]
        power_selection = actions_power[:, :, 1]

        v2v_interference_to_v2i = np.zeros(self.n_sc)
        for i in range(self.n_agent):
            veh_sender = self.agent_to_veh[i]
            for j in range(self.n_neighbor):
                if actions[i][j] != -1:
                    pathloss = (self.v2v_pathloss_to_bs_with_ff[veh_sender]
                                if self.fast_fading_enabled
                                else self.v2v_pathloss_to_bs[veh_sender])
                    v2v_interference_to_v2i[actions[i][j]] += 10 ** (
                        (self.v2v_power_levels_dbm[power_selection[i, j]] - pathloss
                         + self.veh_antenna_gain_dbi + self.bs_antenna_gain_dbi
                         - self.bs_noise_figure_db) / 10)
        self.v2v_interference_to_v2i = v2v_interference_to_v2i + self.noise_power_mw

        pathloss_to_bs = (self.v2i_pathloss_to_bs_with_ff
                          if self.fast_fading_enabled
                          else self.v2i_pathloss_to_bs)
        v2i_signals = 10 ** ((self.v2i_power_dbm - pathloss_to_bs
                              + self.veh_antenna_gain_dbi + self.bs_antenna_gain_dbi
                              - self.bs_noise_figure_db) / 10)

        self.v2i_signals = v2i_signals
        v2i_se = np.log2(1 + np.divide(v2i_signals, self.v2v_interference_to_v2i))
        self.v2i_spectral_efficiency = v2i_se

        total_v2v_interference = np.zeros((self.n_agent, 1))
        v2v_interference_at_rx = np.zeros((self.n_agent, self.n_agent))
        v2v_interference_to_others = np.zeros((self.n_agent, self.n_agent))
        v2i_interference_at_rx = np.zeros((self.n_agent, 1))
        v2v_signal = np.zeros((self.n_agent, 1))

        for i in range(self.n_agent):
            sender = i
            veh_sender = self.agent_to_veh[i]

            for j in range(self.n_neighbor):
                if actions[sender][j] != -1:
                    veh_receiver = self.vehicles_v2v[veh_sender].destinations[0]

                    pathloss_signal = (self.v2v_pathloss_with_ff[veh_sender, veh_receiver]
                                       if self.fast_fading_enabled
                                       else self.v2v_pathloss[veh_sender, veh_receiver])

                    v2v_signal[sender][0] = 10 ** (
                        (self.v2v_power_levels_dbm[power_selection[sender, j]]
                         - pathloss_signal
                         + 2 * self.veh_antenna_gain_dbi - self.veh_noise_figure_db) / 10)

                    pathloss_v2i_to_v2v = (
                        self.v2i_pathloss_to_veh_with_ff[actions[sender][j], veh_receiver]
                        if self.fast_fading_enabled
                        else self.v2i_pathloss_to_veh[actions[sender][j], veh_receiver])

                    v2i_interference_at_rx[sender][0] = 10 ** (
                        (self.v2i_power_dbm - pathloss_v2i_to_v2v
                         + 2 * self.veh_antenna_gain_dbi - self.veh_noise_figure_db) / 10)

                    total_v2v_interference[sender][0] += v2i_interference_at_rx[sender][0]

                    for x in range(self.n_agent):
                        other_sender = x
                        veh_other_sender = self.agent_to_veh[x]

                        if x != i:
                            if actions[sender][j] == actions[other_sender][j]:
                                pathloss_interference = (
                                    self.v2v_pathloss_with_ff[veh_other_sender, veh_receiver]
                                    if self.fast_fading_enabled
                                    else self.v2v_pathloss[veh_other_sender, veh_receiver])

                                v2v_interference_at_rx[sender][other_sender] = 10 ** (
                                    (self.v2v_power_levels_dbm[power_selection[other_sender, j]]
                                     - pathloss_interference
                                     + 2 * self.veh_antenna_gain_dbi - self.veh_noise_figure_db) / 10)

                                total_v2v_interference[sender][0] += v2v_interference_at_rx[sender][other_sender]
                                v2v_interference_to_others[other_sender][sender] = \
                                    v2v_interference_at_rx[sender][other_sender]

        self.total_v2v_interference = total_v2v_interference + self.noise_power_mw
        self.total_v2v_interference_dbm = 10 * np.log10(self.total_v2v_interference)
        v2v_se = np.log2(1 + np.divide(v2v_signal, self.total_v2v_interference))
        self.v2v_spectral_efficiency = v2v_se
        self.v2v_interference_to_others = v2v_interference_to_others
        self.v2v_interference_at_rx = v2v_interference_at_rx

        self.queue -= (v2v_se * self.time_fast * self.bandwidth_hz / self.cam_size_bits)
        self.queue[self.queue <= 0] = 0

        reward_v2v = np.zeros((self.n_agent, 1))
        for i in range(self.n_agent):
            sender = i
            if self.queue[sender] <= 0:
                reward_v2v[sender][0] = 1.0
            else:
                se_loss = np.log2(1 + np.divide(
                    v2v_signal,
                    (self.total_v2v_interference -
                     np.reshape(self.v2v_interference_to_others[sender, :], (-1, 1)))
                )) - v2v_se
                reward_v2v[sender][0] = min(max(
                    (v2v_se[sender][0] - self.reward_lambda1 * np.sum(se_loss)) / 10, -1), 1)

        self.individual_rewards = reward_v2v
        return v2v_se, v2i_se, self.queue, reward_v2v

    # =========================================================================
    # Environment Step
    # =========================================================================

    def step(self, actions, t, interval):
        """Execute one environment step."""
        if self.fast_fading_enabled:
            self._renew_fast_fading()

        action_temp = actions.copy()
        pre_empty_mask = (self.queue <= 0)

        v2v_se, v2i_se, queue, reward_v2v = self._compute_performance_and_reward(action_temp)
        self._compute_interference_per_sc(action_temp[:, :, 0], action_temp[:, :, 1])
        self._renew_active_links()

        if self.task_type == "NFIG":
            done = (t == self.n_step_per_episode - 1)
            global_reward = np.array([[self.v2v_weight * np.sum(v2v_se)]])
            v2i_se = self.v2i_weight * v2i_se
            return global_reward, v2v_se, v2i_se, done

        elif self.task_type == "SIG" or self.task_type == "POSIG":
            individual_rewards = np.zeros_like(self.queue)
            for i in range(self.queue.shape[0]):
                if pre_empty_mask[i][0]:
                    individual_rewards[i][0] = self.reward_queue_empty
                else:
                    individual_rewards[i][0] = v2v_se[i][0] * 0.01 * self.v2v_weight

            done = (t == self.n_step_per_episode - 1)
            global_reward = np.array([[np.sum(individual_rewards) + np.sum(v2i_se) * 0.01 * self.v2i_weight]])


            # print(f"t={t}, v2v_se={v2v_se.flatten()}, queue={self.queue.flatten()}, pre_empty={pre_empty_mask.flatten()}")
            # print(f"individual_rewards={individual_rewards.flatten()}, global_reward={global_reward}")

            
            return global_reward, individual_rewards, v2i_se, done

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    # =========================================================================
    # Episode Management
    # =========================================================================

    def new_random_game(self):
        """Initialize a new episode."""
        self.vehicles_v2v = []
        self.vehicles_v2i = []

        if self.n_veh > 0:
            self._add_vehicles_from_data(self.train_data)
            self._renew_neighbors()
            self._renew_channels()
        else:
            print('Error!!!!')

        self.queue = np.ones((self.n_agent, self.n_neighbor))
        self.active_links = np.ones((self.n_agent, self.n_neighbor), dtype='bool')
        self.previous_interference_per_sc = np.zeros((self.n_agent, self.n_sc))

    def renew_queue(self):
        """Reset queues for new control interval."""
        self.queue = np.ones((self.n_agent, self.n_neighbor))
        for ag_idx in range(self.n_agent):
            if self.queue[ag_idx][0] > self.max_queue_length:
                self.queue[ag_idx][0] = self.max_queue_length

    def _renew_active_links(self):
        """Update which links are still active based on queue status."""
        for ag_idx in range(self.n_agent):
            if self.queue[ag_idx, 0] <= 0:
                self.active_links[ag_idx, 0] = False
            else:
                self.active_links[ag_idx, 0] = True

    def get_available_actions(self):
        """Return available actions for each agent based on link status."""
        available_actions_list = []
        for ag_idx in range(self.n_agent):
            if self.active_links[ag_idx, 0]:
                available_actions = np.ones(self.n_actions)
            else:
                available_actions = np.zeros(self.n_actions)
                available_actions[-1] = 1.0
            available_actions_list.append(available_actions)
        return available_actions_list

    # =========================================================================
    # State Representations
    # =========================================================================

    def get_state(self, idx, epsi, i_step):
        """
        Get state representation based on task type.
        
        - NFIG: _get_simple_state (G_i + G_ji)
        - SIG: _get_global_state (t + G_i + G_ji + G_m + G_Bi + G_iB + I_prev + queue)
        - POSIG: _get_observation (t + G_i + G_iB + I_prev + queue, per agent)
        """
        if self.task_type == "NFIG":
            return self._get_simple_state(idx, epsi, i_step)
        elif self.task_type == "SIG":
            return self._get_global_state(idx, epsi, i_step)
        elif self.task_type == "POSIG":
            return self._get_observation(idx, epsi, i_step)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _get_simple_state(self, idx, epsi, i_step):
        """State for NFIG: G_i + G_ji"""
        state = np.array([])
        g_i = np.array([])
        g_ji = np.array([])

        for i in range(self.n_agent):
            veh_sender = self.agent_to_veh[i]
            veh_receiver = self.vehicles_v2v[veh_sender].destinations[idx[1]]
            channel_norm = np.array([
                self.v2v_pathloss[veh_sender, veh_receiver] / self.norm_v2v_channel_factor
            ])
            g_i = np.hstack((g_i, channel_norm))

        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i == j:
                    continue
                veh_sender = self.agent_to_veh[j]
                veh_receiver = self.vehicles_v2v[self.agent_to_veh[j]].destinations[idx[1]]
                channel_norm = np.array([
                    self.v2v_pathloss[veh_sender, veh_receiver] / self.norm_v2v_channel_factor
                ])
                g_ji = np.hstack((g_ji, channel_norm))

        for state_info in [g_i, g_ji]:
            state = np.hstack((state, state_info))

        state = state.reshape((1, self.state_dim))
        return state

    def _get_global_state(self, idx, epsi, i_step):
        """State for SIG: t + G_i + G_ji + G_m + G_Bi + G_iB + I_prev + queue"""

        def norm_gain(pl_db, gain_type):
            if self.static_norm:
                return pl_db / self.norm_v2v_channel_factor
            else:
                lo_db, hi_db = self.pathloss_bounds[gain_type]
                return np.clip((pl_db - lo_db) / max(hi_db - lo_db, 1e-9), 0.0, 1.0)

        state = np.array([])

        if self.timestep_encoding_type == 'normalized':
            t = np.array([i_step / self.n_step_per_episode])
        else:
            t = np.zeros(self.n_step_per_episode)
            t[min(i_step, self.n_step_per_episode - 1)] = 1

        g_i = np.array([])
        for i in range(self.n_agent):
            veh_sender = self.agent_to_veh[i]
            veh_receiver = self.vehicles_v2v[veh_sender].destinations[idx[1]]
            pl_db = self.v2v_pathloss[veh_sender, veh_receiver]
            g_i = np.hstack((g_i, norm_gain(pl_db, 'v2v_link')))

        g_ji = np.array([])
        for i in range(self.n_agent):
            for j in range(self.n_agent):
                if i == j:
                    continue
                veh_sender = self.agent_to_veh[j]
                veh_receiver = self.vehicles_v2v[self.agent_to_veh[i]].destinations[idx[1]]
                pl_db = self.v2v_pathloss[veh_sender, veh_receiver]
                g_ji = np.hstack((g_ji, norm_gain(pl_db, 'v2v_interference')))

        g_m = np.array([])
        for m in range(self.n_sc):
            pl_db = self.v2i_pathloss_to_bs[m]
            g_m = np.hstack((g_m, norm_gain(pl_db, 'veh_to_bs')))

        g_bi = np.array([])
        for i in range(self.n_agent):
            for m in range(self.n_sc):
                veh_receiver = self.vehicles_v2v[self.agent_to_veh[i]].destinations[idx[1]]
                pl_db = self.v2i_pathloss_to_veh[m, veh_receiver]
                g_bi = np.hstack((g_bi, norm_gain(pl_db, 'v2v_interference')))

        g_ib = np.array([])
        for i in range(self.n_agent):
            veh_sender = self.agent_to_veh[i]
            pl_db = self.v2v_pathloss_to_bs[veh_sender]
            g_ib = np.hstack((g_ib, norm_gain(pl_db, 'veh_to_bs')))

        if not hasattr(self, 'previous_interference_per_sc'):
            self.previous_interference_per_sc = np.zeros((self.n_agent, self.n_sc))

        i_prev = np.array([])
        for agent_idx in range(self.n_agent):
            intf_mw = self.previous_interference_per_sc[agent_idx, :].copy()
            intf_dbm = 10.0 * np.log10(np.maximum(intf_mw, 1e-30))
            noise_dbm = 10.0 * np.log10(self.noise_power_mw)
            inr_db = intf_dbm - noise_dbm
            inr_min, inr_max = self.inr_norm_bounds
            intf_norm = (np.clip(inr_db, inr_min, inr_max) - inr_min) / (inr_max - inr_min)
            i_prev = np.hstack((i_prev, intf_norm))

        queue_norm = self.queue / self.max_queue_length
        queue_norm = queue_norm.flatten()

        for state_info in [t, g_i, g_ji, g_m, g_bi, g_ib, i_prev, queue_norm]:
            state = np.hstack((state, state_info))

        state = state.reshape((1, -1))
        return state

    def _get_observation(self, idx, epsi, i_step):
        """Observation for POSIG: t + G_i + G_iB + I_prev + queue (per agent)"""

        def norm_gain(pl_db, gain_type):
            if self.static_norm:
                return pl_db / self.norm_v2v_channel_factor
            else:
                lo_db, hi_db = self.pathloss_bounds[gain_type]
                return np.clip((pl_db - lo_db) / max(hi_db - lo_db, 1e-9), 0.0, 1.0)

        state = np.array([])
        agent_idx = idx[0]

        if self.timestep_encoding_type == 'normalized':
            t = np.array([i_step / self.n_step_per_episode])
        else:
            t = np.zeros(self.n_step_per_episode)
            t[min(i_step, self.n_step_per_episode - 1)] = 1

        veh_sender = self.agent_to_veh[agent_idx]
        veh_receiver = self.vehicles_v2v[veh_sender].destinations[idx[1]]
        pl_db = self.v2v_pathloss[veh_sender, veh_receiver]
        g_i = np.array([norm_gain(pl_db, 'v2v_link')])

        pl_db = self.v2v_pathloss_to_bs[veh_sender]
        g_ib = np.array([norm_gain(pl_db, 'veh_to_bs')])

        if not hasattr(self, 'previous_interference_per_sc'):
            self.previous_interference_per_sc = np.zeros((self.n_agent, self.n_sc))

        intf_mw = self.previous_interference_per_sc[agent_idx, :].copy()
        intf_dbm = 10.0 * np.log10(np.maximum(intf_mw, 1e-30))
        noise_dbm = 10.0 * np.log10(self.noise_power_mw)
        inr_db = intf_dbm - noise_dbm
        inr_min, inr_max = self.inr_norm_bounds
        i_prev = (np.clip(inr_db, inr_min, inr_max) - inr_min) / (inr_max - inr_min)

        queue_norm = np.array([self.queue[agent_idx, idx[1]] / self.max_queue_length])

        for state_info in [t, g_i, g_ib, i_prev, queue_norm]:
            state = np.hstack((state, state_info))

        state = state.reshape((1, -1))
        return state

    # =========================================================================
    # Interference Computation
    # =========================================================================

    def _compute_interference_per_sc(self, actions, power_selection):
        """Compute interference experienced by each agent on each subchannel."""
        interference_per_sc = np.zeros((self.n_agent, self.n_sc))

        for i in range(self.n_agent):
            veh_receiver_i = self.vehicles_v2v[self.agent_to_veh[i]].destinations[0]

            for m in range(self.n_sc):
                total_interference = self.noise_power_mw

                pathloss = (self.v2i_pathloss_to_veh_with_ff[m, veh_receiver_i]
                            if self.fast_fading_enabled
                            else self.v2i_pathloss_to_veh[m, veh_receiver_i])

                v2i_interference = 10 ** ((self.v2i_power_dbm - pathloss +
                                           2 * self.veh_antenna_gain_dbi - self.veh_noise_figure_db) / 10)
                total_interference += v2i_interference

                for j in range(self.n_agent):
                    if j != i:
                        for neighbor_idx in range(self.n_neighbor):
                            if actions[j][neighbor_idx] == m:
                                veh_sender_j = self.agent_to_veh[j]

                                pathloss = (self.v2v_pathloss_with_ff[veh_sender_j, veh_receiver_i]
                                            if self.fast_fading_enabled
                                            else self.v2v_pathloss[veh_sender_j, veh_receiver_i])

                                v2v_interference = 10 ** ((
                                    self.v2v_power_levels_dbm[power_selection[j, neighbor_idx]] -
                                    pathloss +
                                    2 * self.veh_antenna_gain_dbi - self.veh_noise_figure_db) / 10)

                                total_interference += v2v_interference

                interference_per_sc[i, m] = total_interference

        self.previous_interference_per_sc = interference_per_sc