from types import SimpleNamespace as SN

from Configuration.a2c_params import A2Cparameters
from Configuration.ppo_params import PPOparameters

from Trainers.ia2c_trainer import IA2CTrainer
from Trainers.maa2c_trainer import MAA2CTrainer
from Trainers.ippo_trainer import IPPO_TrainerPS
from Trainers.mappo_trainer import MAPPO_TrainerPS

from Environment.environment_utility import calculate_max_mean_and_ci


class PolicyGradientRunner:
    """
    Unified runner for policy-gradient methods:
      - IA2C / MAA2C
      - IPPO / MAPPO

    algo must be one of: "ia2c", "maa2c", "ippo", "mappo".
    """

    _SUPPORTED = ("ia2c", "maa2c", "ippo", "mappo")

    def __init__(self, env, task_type, env_params, algo: str):
        if algo not in self._SUPPORTED:
            raise ValueError(
                f"PolicyGradientRunner: unsupported algo '{algo}'. "
                f"Use one of {self._SUPPORTED}."
            )
        self.env = env
        self.task_type = task_type
        self.env_params = env_params
        self.algo = algo

    # ---------- param merge ---------- #
    def combine_params(self, algo_params, test_data_list, *, prefer="algo_params"):
        """
        algo_params <-> env_params - single namespace.
        prefer:
          - "algo_params": algo params win on conflict
          - "env_params":  env params win on conflict
        """
        p = SN()
        p.env_params = self.env_params
        p.env = self.env
        p.algo_params = algo_params
        p.task_type = self.task_type

        # 1) copy algorithm parameters
        for k, v in vars(algo_params).items():
            setattr(p, k, v)

        # 2) merge environment parameters
        for k, v in vars(self.env_params).items():
            if hasattr(p, k):
                if prefer == "env_params":
                    setattr(p, k, v)
            else:
                setattr(p, k, v)

        # 3) derived dimensions from env
        p.state_dim = self.env.state_dim
        p.global_state_dim = self.env.global_state_dim
        p.observation_dim = self.env.local_state_dim
        p.action_dim = self.env.n_actions
        p.n_agent = self.env.n_agent

        # 4) test data + trial index
        p.test_data_list = test_data_list
        p.trial_run = 0

        return p

    # ---------- dispatch ---------- #
    def _get_algo_params_and_trainer(self):
        """
        Returns (algo_params_instance, trainer_fn)
        trainer_fn signature: trainer_fn(params) -> (train_rewards, test_rewards)
        """
        if self.algo in ("ia2c", "maa2c"):
            algo_params = A2Cparameters()
            trainer_fn = IA2CTrainer.train_IA2C if self.algo == "ia2c" else MAA2CTrainer.train_MAA2C
            return algo_params, trainer_fn

        if self.algo in ("ippo", "mappo"):
            algo_params = PPOparameters()
            trainer_fn = IPPO_TrainerPS.train_IPPO_ParameterSharing if self.algo == "ippo" else MAPPO_TrainerPS.train_MAPPO_ParameterSharing
            return algo_params, trainer_fn

    # ---------- experiment ---------- #
    def run_experiment(self, test_data_list, *, prefer="algo_params", summarize=True):
        algo_params, trainer_fn = self._get_algo_params_and_trainer()
        params = self.combine_params(algo_params, test_data_list, prefer=prefer)

        test_rewards_n_trials = []

        for trial in range(params.num_trials):
            print(f"[{self.algo.upper()}] Trial: {trial + 1}")
            params.trial_run = trial

            train_rewards, test_rewards = trainer_fn(params)
            test_rewards_n_trials.append(test_rewards)

        if summarize:
            print(self.task_type)
            print("test_rewards_n_trials:", test_rewards_n_trials)

            max_mean, max_mean_ci, mean_over_time, ci_over_time = calculate_max_mean_and_ci(
                test_rewards_n_trials
            )
            print(f"Max Mean: {max_mean}, Confidence Interval: ±{max_mean_ci}")
            print(f"Mean over time: {mean_over_time.tolist()}")
            print(f"Confidence Interval over time: {ci_over_time.tolist()}")

        return test_rewards_n_trials