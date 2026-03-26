from Configuration.idql_params import IDQLparameters
from Trainers.idql_trainer import IDQLtrainerPS, IDQLtrainerNS
from Environment.environment_utility import *


class IDQLrunner:
    def __init__(self, env, env_name, env_params, is_hysteretic_q):
        self.env = env
        self.env_name = env_name
        self.env_params = env_params
        self.is_hysteretic_q = is_hysteretic_q

    def run_experiment(self, test_data_list):
        algo_params = IDQLparameters()

        train_params = {
            'env': self.env,
            'env_name': self.env_name,
            'env_params': self.env_params,
            'test_data_list': test_data_list,
            'is_hysteretic_q': self.is_hysteretic_q,
            'algo_params': algo_params,
        }

        algo_name = "HYS" if self.is_hysteretic_q else "IDQL"
        test_rewards_n_trails = []

        for trial in range(algo_params.num_trials):
            print(f"Trial: {trial + 1}")
            train_rewards, test_rewards = IDQLtrainerNS.train_IDQL_NoSharing(
                **train_params, algo_name=algo_name, trial_run=trial
            )
            test_rewards_n_trails.append(test_rewards)


        print(self.env_name)
        print("test_rewards_n_trails: ", test_rewards_n_trails)
        max_mean, max_mean_ci, mean_over_time, ci_over_time = calculate_max_mean_and_ci(test_rewards_n_trails)
        print(f"Max Mean: {max_mean}, Confidence Interval: ±{max_mean_ci}")
        print(f"Mean over time: {mean_over_time.tolist()}")
        print(f"Confidence Interval over time: {ci_over_time.tolist()}")