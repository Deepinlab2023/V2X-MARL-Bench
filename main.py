import argparse
import os
import time

# Import Environment Information
from Configuration.env_params import V2XParams
from Environment.environment import Environ
from Environment.environment_utility import *

# # Import Runners
# from Runners.ippo_runner import IPPOrunner
# from Runners.a2c_runner import A2Crunner
# from Runners.mappo_runner import MAPPOrunner

from Runners.idql_runner import IDQLrunner
# from Runners.qmix_runner import QMIXrunner




# Utility Functions
# from Results.Plotting.benchmark_plot import benchmark_plot_from_csv, plot_mean_test_returns_comparison

'''
To use this code you must run a script using the terminal
For example:
    NFIG GAME:
    <termina> python main.py --env NFIG --loc 25.0 --algo maa2c
    
    SIG SL GAME:
    <termina> python main.py --env SIG --loc 30.0 --algo maa2c

    SIG ML GAME:
    <termina> python main.py --env SIG --algo maa2c

    POSIG GAME:
    <termina> python main.py --env POSIG --algo maa2c

This will run MAA2C for the HCRA Environment

To Run plotting code use the command:
    <terminal> python main.py --dir_MAA2C Results/MAA2C/random_seeds --dir_MAPPO Results/MAPPO/random_seeds --save_dir Results/Plots

NOTE:
    For specific algorithm configurations, you must look in the config file
    to set algorithm hyperparameters as well as algorithm variation parameters.

    When Plotting, there must be CSV files to get plots, only run the plotting
    script when done running required experiments
'''

def main():
    parser = argparse.ArgumentParser(description="Run different variations of algorithms and environments.")
    parser.add_argument('--env', type=str, help='The environment to run. Choose from "NFIG", "SIG", or "POSIG".')
    parser.add_argument('--loc', type=float, help='The time index for SIG SL and NFIG environments.')
    parser.add_argument('--algo', type=str, help='The algorithm to use. Choose from "ia2c", "ippo", "maa2c", "mappo", "idql", "vdn", "qmix", or "hys".')

    parser.add_argument('--dir_IA2C', type=str, help='Directory containing IA2C CSV files.')
    parser.add_argument('--dir_IPPO', type=str, help='Directory containing IPPO CSV files.')
    parser.add_argument('--dir_MAA2C', type=str, help='Directory containing MAA2C CSV files.')
    parser.add_argument('--dir_MAPPO', type=str, help='Directory containing MAPPO CSV files.')

    parser.add_argument('--test_interval', type=int, default=1000, help='Test interval for plotting.')
    parser.add_argument('--save_dir', type=str, help='Directory to save plot images.')
    args = parser.parse_args()

    test_data_list = []

    # Create the environment
    # More environements can be added here
    if args.env is not None and args.algo is not None:

        # Environment Variable Setup
        env_params = V2XParams(args.env, args.loc)
        env = Environ(env_params)

        # NOTE: renamed per your new convention (veh_data -> train_data)
        train_data = env_params.train_data


        if args.env == "NFIG" and args.loc is not None:  # NFIG
            test_data_list = [sample_veh_position_from_timestep(train_data, args.loc)]
            sampled_data = test_data_list[0]

        elif args.env == "SIG" or args.env == "POSIG":

            if args.loc is None:  # SIG ML / POSIG
                test_data = env_params.test_data
                test_data_list = [loc for _, loc in test_data.groupby("time", sort=False)]

                training_data = env_params.train_data
                sampled_data = sample_veh_positions(1, training_data)

            else:  # SIG SL
                test_data_list = [sample_veh_position_from_timestep(train_data, args.loc)]
                sampled_data = test_data_list[0]

        else:
            raise ValueError("Invalid argument")


        # # Add this after line 78 in main.py (after env = Environ(env_params))
        print("\n" + "="*60)
        print("EXPERIMENT CONFIGURATION")
        print("="*60)
        loc_str = f"_SL_{args.loc}" if args.loc else "_ML"
        print(f"Task:           {env_params.task_type}{loc_str}")
        print(f"Algorithm:      {args.algo}")
        print(f"Num Agents:     {env_params.n_agent}")
        print(f"Train Data:     {getattr(env_params, 'train_data_path', 'N/A')}")
        print(f"Test Data:      {getattr(env_params, 'test_data_path', 'N/A')}")
        print(f"Test Episodes:  {len(test_data_list)}")
        print(f"Steps/Episode:  {env_params.n_step_per_episode}")
        print(f"Fast Fading:    {env_params.fast_fading_tag}")
        print("="*60 + "\n")

        # # NOTE: renamed per your new convention (veh_data -> train_data)
        # env_params.train_data = sampled_data

        if args.env:  # NFIG, SIG, POSIG Game check
            # Create the runner
            # More runners can be added here
            if args.algo == 'ia2c':
                runner = A2Crunner(env, args.env, env_params, algo="ia2c")
            elif args.algo == 'ippo':
                runner = IPPOrunner(env, args.env, env_params)
            elif args.algo == 'maa2c':
                runner = A2Crunner(env, args.env, env_params, algo="maa2c")
            elif args.algo == 'mappo':
                runner = MAPPOrunner(env, args.env, env_params)
            elif args.algo == 'idql':
                runner = IDQLrunner(env, args.env, env_params, False)
            elif args.algo == 'hys':
                runner = IDQLrunner(env, args.env, env_params, True)
            elif args.algo == 'vdn':
                runner = QMIXrunner(env, args.env, env_params, True)
            elif args.algo == 'qmix':
                runner = QMIXrunner(env, args.env, env_params, False)
            else:
                raise ValueError("Algorithm name incorrect or not found")
        else:
            raise ValueError("Environment name incorrect or not found")

        runner.run_experiment(test_data_list)


if __name__ == '__main__':
    start_time = time.time()  # Record the start time
    main()                    # Execute the main function
    end_time = time.time()    # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution Time: {execution_time} seconds")
