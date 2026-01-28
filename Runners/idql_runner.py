from Configuration.idql_params import IDQLparameters
from Trainers.FullyObservable.idql_trainer import IDQLtrainerPS, IDQLtrainerNS
from Trainers.PartialObservable.idql_trainer_po import IDQLtrainerPO
from Envs.UtilityCommunication.veh_position_helper import *


class IDQLrunner:
    def __init__(self, env, env_name, env_params, is_hysteretic_q):
        self.env = env
        self.env_name = env_name
        self.env_params = env_params
        self.is_hysteretic_q = is_hysteretic_q

    def run_experiment(self, test_data_list):
        params = IDQLparameters()

        train_params = {
            'env': self.env,
            'env_name': self.env_name,
            'params': self.env_params,
            'test_data_list': test_data_list,
            # 'veh_pos_data': veh_pos_data,

            # renamed env attributes (old -> new)
            'state_dim': self.env.state_dim,
            'observation_dim': self.env.local_state_dim,
            'action_dim': self.env.n_actions,

            'num_agents': self.env.n_agent,
            'gamma': params.gamma,
            'hidden_dim': params.hidden_dim,
            'value_dim': params.value_dim,
            'tau': params.tau,
            'test_interval': params.test_interval,
            'num_training_iterations': params.num_training_iterations,
            'num_test_episodes': params.num_test_episodes,
            'batch_size': params.batch_size,
            'is_hysteretic_q': self.is_hysteretic_q
        }

        test_rewards_n_trails = []

        if self.env_name == 'POSIG':
            for trial in range(params.num_trials):
                print(f"Trial: {trial + 1}")
                train_rewards, test_rewards = IDQLtrainerPO.train_IDQL_NoSharing(trial, **train_params)
                test_rewards_n_trails.append(test_rewards)
        else:
            for trial in range(params.num_trials):
                print(f"Trial: {trial + 1}")
                if params.no_sharing:
                    train_rewards, test_rewards = IDQLtrainerNS.train_IDQL_NoSharing(trial, **train_params)
                    test_rewards_n_trails.append(test_rewards)
                else:
                    # not yet finished
                    train_rewards, test_rewards = IDQLtrainerPS.train_IDQL_ParameterSharing(trial, **train_params)

        print(self.env_name)
        print("test_rewards_n_trails: ", test_rewards_n_trails)
        max_mean, max_mean_ci, mean_over_time, ci_over_time = calculate_max_mean_and_ci(test_rewards_n_trails)
        # Output results
        print(f"Max Mean: {max_mean}, Confidence Interval: ±{max_mean_ci}")
        print(f"Mean over time: {mean_over_time.tolist()}")
        print(f"Confidence Interval over time: {ci_over_time.tolist()}")
