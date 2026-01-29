from Configuration.qmix_params import QMIXparameters
from Trainers.FullyObservable.qmix_trainer import QMIXtrainerPS, QMIXtrainerNS
from Envs.UtilityCommunication.veh_position_helper import *
from Trainers.PartialObservable.qmix_trainer_po import QMIXtrainerPO


class QMIXrunner:
    def __init__(self, env, env_name, env_params, is_vdn):
        self.env = env
        self.env_name = env_name
        self.env_params = env_params
        self.is_vdn = is_vdn
        
    def run_experiment(self, test_data_list):
        params = QMIXparameters()

        train_params = {
            'env': self.env,
            'env_name': self.env_name,
            'params': self.env_params,
            'test_data_list': test_data_list,
            # 'veh_pos_data': veh_pos_data,
            'state_dim': self.env.stateDim,
            'observation_dim': self.env.local_stateDim,
            'action_dim': self.env.actionDim,
            'num_agents': self.env.n_agent,
            'gamma': params.gamma,
            'hidden_dim': params.hidden_dim,
            'value_dim': params.value_dim,
            'tau': params.tau,
            'test_interval':params.test_interval,
            'num_training_iterations': params.num_training_iterations,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.batch_size,
            'is_vdn': self.is_vdn
        }

        test_rewards_n_trails = []

        # if params.partial:
        if self.env_name == 'POSIG':

            for trial in range(params.num_trials):
                print(f"Trial: {trial+1}")
                train_rewards, test_rewards = QMIXtrainerPO.train_QMIX_NoSharing(trial, **train_params)
                test_rewards_n_trails.append(test_rewards)
        else:

            for trial in range(params.num_trials):
                print(f"Trial: {trial+1}")
                if params.no_sharing:
                    train_rewards, test_rewards = QMIXtrainerNS.train_QMIX_NoSharing(trial, **train_params)
                    test_rewards_n_trails.append(test_rewards)
                    
                else:
                    # not yet finished
                    train_rewards, test_rewards = VDNtrainerPS.train_QMIX_ParameterSharing(trial, **train_params)

        print(self.env_name)
        print("test_rewards_n_trails: ", test_rewards_n_trails)
        max_mean, max_mean_ci, mean_over_time, ci_over_time = calculate_max_mean_and_ci(test_rewards_n_trails)
        # Output results
        print(f"Max Mean: {max_mean}, Confidence Interval: ±{max_mean_ci}")
        print(f"Mean over time: {mean_over_time.tolist()}")
        print(f"Confidence Interval over time: {ci_over_time.tolist()}")

