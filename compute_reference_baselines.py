"""
Standalone script to compute G^min (random baseline) and an approximate G^max
(greedy baseline) for a given location/task -- the two reference points the
paper's normalization formula needs:

    norm_G_t^a = (G_t^a - G_t^min) / (G_t^max - G_t^min)

G^max here is APPROXIMATE, not the true optimum: the true exhaustive/optimal
baseline (Helpers/stat_helper.py: compute_exhaustive_baseline) is combinatorially
intractable for 16 agents (up to 13^16 joint actions per timestep -- its own
docstring says it's only feasible for ~4 agents). The greedy baseline
(top-k candidate search + local improvement passes) is used instead as a
practical stand-in for a high-performance reference.

This does NOT train anything -- it just runs the two baseline policies
through the environment and reports their returns.

Usage:
    python compute_reference_baselines.py --loc 0.0 --tag NFF --n_agent 16
    python compute_reference_baselines.py --loc 3.0 --tag FF --n_agent 16 --random_trials 20
"""
import argparse

from Configuration.env_params import V2XParams
from Environment.environment import Environ
from Environment.environment_utility import sample_veh_position_from_timestep
from Helpers.stat_helper import compute_random_baseline, compute_greedy_baseline


def main():
    parser = argparse.ArgumentParser(description="Compute G^min/G^max reference baselines for SIG SL.")
    parser.add_argument("--loc", type=float, required=True, help="Location index (0.0-8.0)")
    parser.add_argument("--tag", choices=["NFF", "FF"], default="NFF", help="Fast fading on/off")
    parser.add_argument("--n_agent", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument("--random_trials", type=int, default=10,
                         help="Rollouts for the random (G^min) baseline")
    parser.add_argument("--greedy_top_k", type=int, default=5,
                         help="Candidate actions per agent for the greedy (approx G^max) baseline")
    parser.add_argument("--greedy_local_search_passes", type=int, default=1)
    args = parser.parse_args()

    env_params = V2XParams("SIG", args.loc)
    env_params.n_agent = args.n_agent
    env_params.n_veh_per_platoon = [2] * args.n_agent
    env_params.n_veh = 2 * args.n_agent
    env_params._load_vehicle_data()
    env_params.agent_to_veh = env_params._build_agent_to_veh_mapping()
    env_params.fast_fading_enabled = (args.tag == "FF")
    env_params.fast_fading_tag = args.tag

    env = Environ(env_params)
    train_data = env_params.train_data
    test_data_list = [sample_veh_position_from_timestep(train_data, args.loc)]

    print("=" * 60)
    print(f"Reference baselines: loc={args.loc}, tag={args.tag}, n_agent={args.n_agent}")
    print("=" * 60)

    print("\n--- G^min: random baseline ---")
    _, avg_random_return = compute_random_baseline(
        env, test_data_list, num_trials_per_location=args.random_trials
    )
    print(f"\nG^min (avg random return) = {avg_random_return}")

    print("\n--- G^max (approx): greedy baseline ---")
    _, avg_greedy_return = compute_greedy_baseline(
        env, test_data_list, num_trials_per_location=1,
        top_k_per_agent=args.greedy_top_k,
        local_search_passes=args.greedy_local_search_passes,
    )
    print(f"\nG^max_approx (avg greedy return) = {avg_greedy_return}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"G^min          = {avg_random_return}")
    print(f"G^max (approx) = {avg_greedy_return}")
    print(f"Range          = {avg_greedy_return - avg_random_return}")


if __name__ == "__main__":
    main()
