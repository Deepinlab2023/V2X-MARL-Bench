from types import SimpleNamespace as SN
from Configuration.a2c_params import A2Cparameters

from Trainers.ia2c_trainer import IA2CTrainer
# from Trainers.maa2c_trainer import MAA2CTrainer


class A2Crunner:
    """
    Unified runner for IA2C and MAA2C.

    algo must be either "ia2c" or "maa2c".
    """
    def __init__(self, env, task_type, env_params, algo: str):
        if algo not in ("ia2c", "maa2c"):
            raise ValueError(f"A2Crunner: unsupported algo '{algo}'. "
                             f"Use 'ia2c' or 'maa2c'.")
        self.env = env
        self.task_type = task_type
        self.env_params = env_params
        self.algo = algo

    # ---------- param merge ---------- #
    def combine_params(self, algo_params, test_data_list, *, prefer="algo_params"):
        """
        Merge algo_params <-> env_params into a single namespace.
        prefer:
          - "algo_params": algo params win on conflict (current behavior)
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

    # ---------- experiment ---------- #
    def run_experiment(self, test_data_list):
        # pick the right param class + trainer based on algo
        algo_params = A2Cparameters()
        if self.algo == "ia2c":
            trainer_fn = IA2CTrainer.train_IA2C
        elif self.algo == "maa2c":
            trainer_fn = MAA2CTrainer.train_MAA2C

        params = self.combine_params(algo_params, test_data_list, prefer="algo_params")

        test_rewards_n_trials = []

        for trial in range(params.num_trials):
            print(f"[{self.algo.upper()}] Trial: {trial + 1}")
            params.trial_run = trial

            train_rewards, test_rewards = trainer_fn(params)
            test_rewards_n_trials.append(test_rewards)

        # if you ever want them:
        # return test_rewards_n_trials