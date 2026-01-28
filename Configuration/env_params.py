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
DECORRELATION_DISTANCE_M = 10
SHADOW_STD_DB = 3
NOISE_POWER_DBM = -114

# Power levels
V2V_POWER_LEVELS_DBM = [23, 15, 5]
V2I_POWER_DBM = 23

# CAM message (ETSI)
CAM_SIZE_BITS = 8 * 1060 * 6

# Bandwidth
BANDWIDTH_PER_SC_HZ = 1_000_000  # 1 MHz

# Normalization factors
NORM_V2V_CHANNEL_FACTOR = 120
NORM_V2V_INTERFERENCE_FACTOR = 80


# =============================================================================
# Data File Paths
# =============================================================================

DATA_PATHS = {
    ("NFIG", 4): './Environment/SUMOData/NFIG_4ag.csv',
    ("NFIG", 8): './Environment/SUMOData/NFIG_8ag.csv',
    ("NFIG", 16): './Environment/SUMOData/NFIG_16ag.csv',
    ("SIG_SL", 4): './Environment/SUMOData/NFIG_4ag.csv',
    ("SIG_SL", 8): './Environment/SUMOData/NFIG_8ag.csv',
    ("SIG_SL", 16): './Environment/SUMOData/NFIG_16ag.csv',
    ("SIG_ML", 4): './Environment/SUMOData/SIG_ML_4ag_15k.csv',
    ("SIG_ML", 8): './Environment/SUMOData/SIG_ML_8ag_60k.csv',
    ("SIG_ML", 16): './Environment/SUMOData/SIG_ML_16ag_60k.csv',
    ("POSIG", 4): './Environment/SUMOData/SIG_ML_4ag_15k.csv',
    ("POSIG", 8): './Environment/SUMOData/SIG_ML_8ag_60k.csv',
    ("POSIG", 16): './Environment/SUMOData/SIG_ML_16ag_60k.csv',
}
TEST_DATA_PATHS = {
    4: './Environment/SUMOData/NFIG_4ag.csv',
    8: './Environment/SUMOData/NFIG_8ag.csv',
    16: './Environment/SUMOData/NFIG_16ag.csv',
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
        self.n_neighbor = 1

        # =====================================================================
        # Physical Parameters (from constants)
        # =====================================================================
        self.bs_antenna_gain_dbi = BS_ANTENNA_GAIN_DBI
        self.bs_noise_figure_db = BS_NOISE_FIGURE_DB
        self.veh_antenna_gain_dbi = VEH_ANTENNA_GAIN_DBI
        self.veh_noise_figure_db = VEH_NOISE_FIGURE_DB
        self.decorrelation_distance_m = DECORRELATION_DISTANCE_M
        self.shadow_std_db = SHADOW_STD_DB
        self.noise_power_dbm = NOISE_POWER_DBM
        self.v2v_power_levels_dbm = V2V_POWER_LEVELS_DBM
        self.v2i_power_dbm = V2I_POWER_DBM
        self.cam_size_bits = CAM_SIZE_BITS
        self.bandwidth_per_sc_hz = BANDWIDTH_PER_SC_HZ
        self.norm_v2v_channel_factor = NORM_V2V_CHANNEL_FACTOR
        self.norm_v2v_interference_factor = NORM_V2V_INTERFERENCE_FACTOR

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
        # Sampling Configuration
        # =====================================================================
        self.sample_method = 'random'  # Options: 'random', 'consecutive', 'ordered'
        self.sampling_size = 1

        # =====================================================================
        # Training Configuration
        # =====================================================================
        self.n_episodes = 50000
        self.t_max = self.n_step_per_episode
        self.n_episodes_test = 2
        self.max_queue_length = 1
        self.seed = 1

        # =====================================================================
        # Reward Configuration
        # =====================================================================
        self.reward_lambda1 = 0.001
        self.reward_lambda2 = 0.1
        self.reward_lambda3 = 1
        self.reward_queue_empty = 0.5

        # =====================================================================
        # Fast Fading
        # =====================================================================
        self.fast_fading_enabled = False
        self.fast_fading_tag = "FF" if self.fast_fading_enabled else "NFF"

        # =====================================================================
        # Action Space
        # =====================================================================
        self.state_type = 'normal_version'
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