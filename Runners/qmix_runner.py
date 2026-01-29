from Configuration.qmix_params import QMIXparameters
from Trainers.qmix_trainer import QMIXtrainerPS, QMIXtrainerNS
from Environment.environment_utility import *



class QMIXrunner:
    def __init__(self, env, env_name, env_params, is_vdn):
        self.env = env
        self.env_name = env_name
        self.env_params = env_params
        self.is_vdn = is_vdn

    def run_experiment(self, test_data_list):
        algo_params = QMIXparameters()

        train_params = {
            'env': self.env,
            'env_name': self.env_name,
            'env_params': self.env_params,
            'test_data_list': test_data_list,
            'is_vdn': self.is_vdn,
            'algo_params': algo_params,
        }

        test_rewards_n_trails = []


        for trial in range(algo_params.num_trials):
            print(f"Trial: {trial + 1}")
            train_rewards, test_rewards = QMIXtrainerNS.train_QMIX_NoSharing(**train_params)
            test_rewards_n_trails.append(test_rewards)


        print(self.env_name)
        print("test_rewards_n_trails: ", test_rewards_n_trails)
        max_mean, max_mean_ci, mean_over_time, ci_over_time = calculate_max_mean_and_ci(test_rewards_n_trails)
        print(f"Max Mean: {max_mean}, Confidence Interval: ±{max_mean_ci}")
        print(f"Mean over time: {mean_over_time.tolist()}")
        print(f"Confidence Interval over time: {ci_over_time.tolist()}")