import argparse
import os
import random
import time

import numpy as np
import torch as th

# Import Environment Information
from Configuration.env_params import V2XParams
from Configuration.param_loader import resolve_param_overrides
from Environment.environment import Environ
from Environment.environment_utility import *

# Import Runners
from Runners.policy_gradient_runner import PolicyGradientRunner

from Runners.idql_runner import IDQLrunner
from Runners.qmix_runner import QMIXrunner




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
def get_task_label(env_name: str, loc):
    if env_name == "NFIG":
        if loc is None:
            raise ValueError("NFIG requires --loc (time index).")
        return "NFIG"

    if env_name == "SIG":
        return "SIG SL" if loc is not None else "SIG ML"

    if env_name == "POSIG":
        if loc is not None:
            raise ValueError("POSIG must be run with --loc omitted (ML setting).")
        return "POSIG"

    raise ValueError(f"Unknown env: {env_name}")


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
    parser.add_argument('--seed', type=int, default=None, help='Global random seed for reproducibility.')
    parser.add_argument('--n_agent', type=int, default=None, choices=[4, 8, 16],
                        help='Override number of agents (default: from env_params).')
    parser.add_argument('--train_data', type=str, default=None,
                        help='Override training data CSV path.')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Override test data CSV path.')
    parser.add_argument('--config', type=str, default=None,
                        help='Sparse JSON file overriding preset parameters (only changed fields needed).')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(args.seed)

    test_data_list = []

    # Create the environment
    # More environements can be added here
    if args.env is not None and args.algo is not None:

        # Environment Variable Setup
        env_params = V2XParams(args.env, args.loc)
        if args.n_agent is not None:
            env_params.n_agent = args.n_agent
            env_params.n_veh_per_platoon = [2] * args.n_agent
            env_params.n_veh = 2 * args.n_agent
            if args.train_data is None:
                env_params._load_vehicle_data()
            env_params.agent_to_veh = env_params._build_agent_to_veh_mapping()
        if args.train_data is not None:
            env_params.train_data = load_veh_pos(args.train_data)
            env_params.train_data_path = args.train_data
        if args.test_data is not None:
            env_params.test_data = load_veh_pos(args.test_data)
            env_params.test_data_path = args.test_data
        env = Environ(env_params)

        # NOTE: renamed per your new convention (veh_data -> train_data)
        train_data = env_params.train_data


        if args.env == "NFIG" and args.loc is not None:  # NFIG
            test_data_list = [sample_veh_position_from_timestep(train_data, args.loc)]
            sampled_data = test_data_list[0]

        elif args.env == "SIG" or args.env == "POSIG":

            if args.loc is None:  # SIG ML / POSIG
                test_data = env_params.test_data
                test_data_list = [loc for _, loc in test_data.groupby("snapshot_id", sort=False)]

                training_data = env_params.train_data
                sampled_data = sample_veh_positions(1, training_data)

            else:  # SIG SL
                test_data_list = [sample_veh_position_from_timestep(train_data, args.loc)]
                sampled_data = test_data_list[0]

        else:
            raise ValueError("Invalid argument")


        # # # Add this after line 78 in main.py (after env = Environ(env_params))
        # print("\n" + "="*60)
        # print("EXPERIMENT CONFIGURATION")
        # print("="*60)
        # loc_str = f"_SL_{args.loc}" if args.loc else "_ML"
        # print(f"Task:           {env_params.task_type}{loc_str}")
        # print(f"Algorithm:      {args.algo}")
        # print(f"Num Agents:     {env_params.n_agent}")
        # print(f"Train Data:     {getattr(env_params, 'train_data_path', 'N/A')}")
        # print(f"Test Data:      {getattr(env_params, 'test_data_path', 'N/A')}")
        # print(f"Test Episodes:  {len(test_data_list)}")
        # print(f"Time Horizon:   {env_params.n_step_per_episode}")
        # print(f"Fast Fading:    {env_params.fast_fading_tag}")
        # print("="*60 + "\n")
        task_label = get_task_label(args.env, args.loc)

        param_overrides, preset_path, _ = resolve_param_overrides(
            args.algo, args.env, args.loc, args.config
        )

        print("\n" + "="*60)
        print("EXPERIMENT CONFIGURATION")
        print("="*60)
        print(f"Task:           {task_label}")
        print(f"Algorithm:      {args.algo}")
        print(f"Preset:         {preset_path if preset_path else 'none (class defaults)'}")
        if args.config:
            print(f"User Config:    {args.config}")
        print(f"Num Agents:     {env_params.n_agent}")
        print(f"Train Data:     {getattr(env_params, 'train_data_path', 'N/A')}")
        print(f"Test Data:      {getattr(env_params, 'test_data_path', 'N/A')}")
        print(f"Test Episodes:  {len(test_data_list)}")
        print(f"Steps/Episode:  {env_params.n_step_per_episode}")
        print(f"Fast Fading:    {env_params.fast_fading_tag}")
        print(f"State Dim:      {env.state_dim}")
        print("="*60 + "\n")

        

        # # NOTE: renamed per your new convention (veh_data -> train_data)
        # env_params.train_data = sampled_data

        if args.env:  # NFIG, SIG, POSIG Game check
            # Create the runner
            # More runners can be added here
            if args.algo in ("ia2c", "maa2c", "ippo", "mappo"):
                runner = PolicyGradientRunner(env, args.env, env_params, algo=args.algo, param_overrides=param_overrides)
            elif args.algo == 'idql':
                runner = IDQLrunner(env, args.env, env_params, False, param_overrides=param_overrides)
            elif args.algo == 'hys':
                runner = IDQLrunner(env, args.env, env_params, True, param_overrides=param_overrides)
            elif args.algo == 'vdn':
                runner = QMIXrunner(env, args.env, env_params, True, param_overrides=param_overrides)
            elif args.algo == 'qmix':
                runner = QMIXrunner(env, args.env, env_params, False, param_overrides=param_overrides)
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
