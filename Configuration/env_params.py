"""
V2X Environment Parameters for Multi-Agent Reinforcement Learning.

Contains configuration for C-V2X resource allocation experiments.
Physical parameters follow 3GPP TR 36.885 and ETSI standards.
"""

from Environment.environment_utility import load_veh_pos


# =============================================================================
# Physical Constants (3GPP TR 36.885, ETSI EN 302 637-2)
# =============================================================================

# Antenna parameters
BS_ANTENNA_GAIN_DBI = 8
BS_NOISE_FIGURE_DB = 5
VEH_ANTENNA_GAIN_DBI = 3
VEH_NOISE_FIGURE_DB = 9

# Channel model
CARRIER_FREQUENCY_GHZ = 2
NOISE_POWER_DBM = -114

# Antenna heights (m)
H_TX_M = 1.5  # V2V transmitter antenna height
H_RX_M = 1.5  # V2V receiver antenna height
BS_ANTENNA_HEIGHT_M = 25  # Base station antenna height

# Power levels
V2V_POWER_LEVELS_DBM = [23, 10, 5]
V2I_POWER_DBM = 23

# CAM message (ETSI)
CAM_SIZE_BITS = 8 * 1060 * 6

# Bandwidth
BANDWIDTH_PER_SC_HZ = 1_000_000  # 1 MHz

# Timing
TIME_FAST_S = 0.001  # Fast fading update interval (1 ms)

# Normalization factors
NORM_V2V_CHANNEL_FACTOR = 120

# =============================================================================
# Topology Configuration
# =============================================================================

BS_POSITION = [0, -35]  # Base station position [x, y] in meters

# =============================================================================
# Normalization Bounds (empirically determined)
# =============================================================================

PATHLOSS_BOUNDS = {
    "v2v_link": (44.5, 119.3),
    "v2v_interference": (44.5, 125.6),
    "veh_to_bs": (76.9, 128.1),
}
INR_NORM_BOUNDS = (20.0, 85.0)  # INR normalization range in dB

# =============================================================================
# State Encoding Configuration
# =============================================================================

TIMESTEP_ENCODING_TYPE = 'one-hot'  # Options: 'one-hot', 'normalized'

# =============================================================================
# Data File Paths
# =============================================================================

DATA_PATHS = {
    ("NFIG", 4): './Environment/SUMOData/NFIG_k4.csv',
    ("NFIG", 8): './Environment/SUMOData/NFIG_k8.csv',
    ("NFIG", 16): './Environment/SUMOData/NFIG_k16.csv',
    ("SIG_SL", 4): './Environment/SUMOData/NFIG_k4.csv',
    ("SIG_SL", 8): './Environment/SUMOData/NFIG_k8.csv',
    ("SIG_SL", 16): './Environment/SUMOData/NFIG_k16.csv',
    ("SIG_ML", 4): './Environment/SUMOData/SIG_ML_k4.csv',
    ("SIG_ML", 8): './Environment/SUMOData/SIG_ML_k8.csv',
    ("SIG_ML", 16): './Environment/SUMOData/SIG_ML_k16.csv',
    ("POSIG", 4): './Environment/SUMOData/SIG_ML_k4.csv',
    ("POSIG", 8): './Environment/SUMOData/SIG_ML_k8.csv',
    ("POSIG", 16): './Environment/SUMOData/SIG_ML_k16.csv',
}
TEST_DATA_PATHS = {
    4: './Environment/SUMOData/NFIG_k4.csv',
    8: './Environment/SUMOData/NFIG_k8.csv',
    16: './Environment/SUMOData/NFIG_k16.csv',
}


class V2XParams:
    """
    Configuration parameters for V2X resource allocation environment.

    Attributes:
        task_type: Task type ('NFIG', 'SIG', 'POSIG')
        n_agent: Number of V2V agents (4, 8, or 16)
        loc: Location identifier for single-location SIG experiments (SIG_SL)
             If None, uses multi-location data (SIG_ML)
    """

    def __init__(self, task_type, loc=None):
        self.task_type = task_type
        self.loc = loc

        # =====================================================================
        # Network Topology
        # =====================================================================
        self.n_agent = 4
        self.n_sc = 4
        self.n_veh_per_platoon = [2] * self.n_agent
        self.n_veh = sum(self.n_veh_per_platoon)

        # =====================================================================
        # Physical Parameters (from constants)
        # =====================================================================
        self.bs_antenna_gain_dbi = BS_ANTENNA_GAIN_DBI
        self.bs_noise_figure_db = BS_NOISE_FIGURE_DB
        self.veh_antenna_gain_dbi = VEH_ANTENNA_GAIN_DBI
        self.veh_noise_figure_db = VEH_NOISE_FIGURE_DB
        self.noise_power_dbm = NOISE_POWER_DBM
        self.v2v_power_levels_dbm = V2V_POWER_LEVELS_DBM
        self.v2i_power_dbm = V2I_POWER_DBM
        self.cam_size_bits = CAM_SIZE_BITS
        self.bandwidth_per_sc_hz = BANDWIDTH_PER_SC_HZ
        self.norm_v2v_channel_factor = NORM_V2V_CHANNEL_FACTOR

        # Channel model parameters
        self.carrier_freq_ghz = CARRIER_FREQUENCY_GHZ
        self.h_tx_m = H_TX_M
        self.h_rx_m = H_RX_M
        self.bs_antenna_height_m = BS_ANTENNA_HEIGHT_M
        self.bs_position = BS_POSITION
        self.time_fast_s = TIME_FAST_S

        # Normalization bounds
        self.pathloss_bounds = PATHLOSS_BOUNDS
        self.inr_norm_bounds = INR_NORM_BOUNDS

        # State encoding
        self.timestep_encoding_type = TIMESTEP_ENCODING_TYPE

        # =====================================================================
        # Task-Specific Configuration
        # =====================================================================
        if self.task_type == "NFIG":
            self.n_step_per_episode = 1
            self.v2v_weight = 0.9
            self.v2i_weight = 0.1
        else:  # SIG or POSIG
            self.n_step_per_episode = 50
            self.v2v_weight = 1.8
            self.v2i_weight = 0.2

        # =====================================================================
        # Reward Configuration
        # =====================================================================
        self.max_queue_length = 1
        self.reward_queue_empty = 0.5

        # =====================================================================
        # Fast Fading
        # =====================================================================
        self.fast_fading_enabled = True
        self.fast_fading_tag = "FF" if self.fast_fading_enabled else "NFF"

        # =====================================================================
        # Action Space
        # =====================================================================
        self.n_power_levels = len(self.v2v_power_levels_dbm)
        self.n_actions = self.n_power_levels * self.n_sc + 1

        # =====================================================================
        # Data Loading
        # =====================================================================
        self.train_data = None
        self.test_data = None
        self._load_vehicle_data()

        # =====================================================================
        # Agent to Vehicle Mapping
        # =====================================================================
        self.agent_to_veh = self._build_agent_to_veh_mapping()

    def _get_data_key(self):
        """Determine the data path key based on task_type and loc."""
        if self.task_type == "NFIG":
            return ("NFIG", self.n_agent)
        elif self.task_type == "SIG" and self.loc is not None:
            return ("SIG_SL", self.n_agent)
        elif self.task_type == "SIG" and self.loc is None:
            return ("SIG_ML", self.n_agent)
        elif self.task_type == "POSIG":
            return ("POSIG", self.n_agent)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def _load_vehicle_data(self):
        """Load training and test vehicle position data."""
        data_key = self._get_data_key()

        if data_key not in DATA_PATHS:
            raise ValueError(f"No data path configured for {data_key}")

        self.train_data_path = DATA_PATHS[data_key]
        self.test_data_path = TEST_DATA_PATHS[self.n_agent]

        self.train_data = load_veh_pos(self.train_data_path)
        self.test_data = load_veh_pos(self.test_data_path)

    def _build_agent_to_veh_mapping(self):
        """Build mapping from agent index to vehicle index."""
        agent_to_veh = {}
        agent_idx = 0
        veh_idx = 0

        for platoon_size in self.n_veh_per_platoon:
            agent_to_veh[agent_idx] = veh_idx
            veh_idx += platoon_size
            agent_idx += 1

        return agent_to_veh