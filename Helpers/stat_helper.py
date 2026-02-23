
import numpy as np
import copy
from itertools import product
from typing import List, Tuple, Optional


def compute_random_baseline(
    env,
    test_data_list: List,
    *,
    num_trials_per_location: int = 10,
    horizon: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[List[float]], float]:
    """
    Compute random action baseline across all test locations.
    
    Args:
        env: The V2X environment instance
        test_data_list: List of location data (one per test topology)
        num_trials_per_location: Number of random rollouts per location
        horizon: Episode length (defaults to env.n_step if available, else 50)
        verbose: Whether to print detailed results
    
    Returns:
        all_location_returns: List of lists, [n_locations][n_trials] returns
        avg_random_return: Average return across all trials and locations
    """
    num_agents = env.n_agent
    action_dim = env.n_actions
    horizon = horizon or getattr(env, 'n_step', getattr(env, 'n_step_per_episode', 50))
    
    all_location_returns: List[List[float]] = []
    
    for loc_idx, location_data in enumerate(test_data_list):
        # CRITICAL: Use train_data, not loaded_veh_data
        env.train_data = location_data
        
        location_returns: List[float] = []
        for trial in range(num_trials_per_location):
            env.new_random_game()
            
            episode_return = 0.0
            for t in range(horizon):
                # Random joint action
                RRA = np.zeros((num_agents, 1, 2), dtype=np.int32)
                for ag in range(num_agents):
                    action = np.random.randint(0, action_dim)
                    sc, pw = env.map_action_to_rra(action, agent_idx=ag)
                    RRA[ag, 0, 0] = sc
                    RRA[ag, 0, 1] = pw
                
                global_reward, _ = env.step(RRA.copy(), t)
                episode_return += float(np.asarray(global_reward).reshape(-1)[0])
            
            location_returns.append(episode_return)
        
        all_location_returns.append(location_returns)
        
        if verbose:
            print(f"Location {loc_idx}: mean={np.mean(location_returns):.3f}, "
                  f"std={np.std(location_returns):.3f}")
    
    # Flatten and compute overall statistics
    all_returns_flat = [r for loc in all_location_returns for r in loc]
    avg_random_return = float(np.mean(all_returns_flat))
    
    if verbose:
        _print_baseline_summary("Random", all_location_returns, num_trials_per_location, 
                                avg_random_return, all_returns_flat)
    
    return all_location_returns, avg_random_return


def compute_greedy_baseline(
    env,
    test_data_list: List,
    *,
    num_trials_per_location: int = 1,
    horizon: Optional[int] = None,
    top_k_per_agent: int = 5,
    local_search_passes: int = 1,
    verbose: bool = True,
) -> Tuple[List[List[float]], float]:
    """
    Compute greedy planner baseline across all test locations.
    
    Args:
        env: The V2X environment instance
        test_data_list: List of location data (one per test topology)
        num_trials_per_location: Number of rollouts per location
        horizon: Episode length (defaults to env.n_step if available, else 50)
        top_k_per_agent: Candidate actions per agent for greedy search
        local_search_passes: Number of local improvement passes
        verbose: Whether to print detailed results
    
    Returns:
        all_location_returns: List of lists, [n_locations][n_trials] returns
        avg_greedy_return: Average return across all trials and locations
    """
    num_agents = env.n_agent
    action_dim = env.n_actions
    horizon = horizon or getattr(env, 'n_step', getattr(env, 'n_step_per_episode', 50))
    
    # --- Build action-to-RRA mapping once ---
    action_to_rra: List[Tuple[int, int]] = []
    silent_action_idx = action_dim - 1  # default
    
    for a in range(action_dim):
        sc, pw = env.map_action_to_rra(a, agent_idx=0)
        sc, pw = int(sc), int(pw)
        action_to_rra.append((sc, pw))
        if (sc, pw) == (-1, -1):
            silent_action_idx = a
    
    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        RRA = np.zeros((num_agents, 1, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA
    
    def flatten_q(qarr) -> np.ndarray:
        return np.array(qarr, dtype=float).reshape(-1)
    
    def queues_empty(qarr) -> bool:
        return np.all(flatten_q(qarr) <= 0.0)
    
    def run_greedy_episode(base_env) -> float:
        """
        Run one greedy episode.
        Uses deepcopy to avoid modifying base_env.
        """
        
        # Prepass: compute solo service per agent×action
        prepass_env = copy.deepcopy(base_env)
        base_q = flatten_q(prepass_env.queue)
        per_action_service = [np.zeros(action_dim, dtype=float) for _ in range(num_agents)]
        per_agent_best = np.zeros(num_agents, dtype=float)
        
        for ag in range(num_agents):
            best = 0.0
            silent_joint = [silent_action_idx] * num_agents
            for a in range(action_dim):
                joint = silent_joint.copy()
                joint[ag] = a
                tmp = copy.deepcopy(prepass_env)
                RRA = rra_from_joint_actions(joint)
                tmp.step(RRA.copy(), 0)
                after_q = flatten_q(tmp.queue)
                served = float(max(0.0, base_q[ag] - after_q[ag]))
                per_action_service[ag][a] = served
                if served > best:
                    best = served
            per_agent_best[ag] = best
        
        # Best action per subchannel
        best_action_per_sc: List[dict] = []
        for ag in range(num_agents):
            per_sc = {}
            for a in range(action_dim):
                sc, _ = action_to_rra[a]
                if sc == -1:
                    continue
                val = per_action_service[ag][a]
                if sc not in per_sc or val > per_sc[sc][1]:
                    per_sc[sc] = (a, val)
            best_action_per_sc.append(per_sc)
        
        def candidate_actions_for_agent(ag: int, q_i: float) -> List[int]:
            if q_i <= 0.0:
                return [silent_action_idx]
            cands = [(a, v) for (a, v) in [best_action_per_sc[ag][sc] for sc in best_action_per_sc[ag]]]
            finish_flag = [(a, int(v + 1e-12 >= q_i), v) for (a, v) in cands]
            finish_flag.sort(key=lambda x: (x[1], x[2]), reverse=True)
            ranked = [a for (a, _, _) in finish_flag][:top_k_per_agent]
            if silent_action_idx not in ranked:
                ranked.append(silent_action_idx)
            return ranked
        
        # Main loop
        run_env = copy.deepcopy(base_env)
        per_step_rewards: List[float] = []
        
        for t in range(horizon):
            qnow = flatten_q(run_env.queue)
            
            if queues_empty(qnow):
                actions = [silent_action_idx] * num_agents
                RRA = rra_from_joint_actions(actions)
                global_reward, _ = run_env.step(RRA.copy(), t)
                per_step_rewards.append(float(np.asarray(global_reward).reshape(-1)[0]))
                continue
            
            # Build candidate lists
            cand_lists = [candidate_actions_for_agent(ag, qnow[ag]) for ag in range(num_agents)]
            
            # Channel-aware greedy initialization
            order = []
            for ag in range(num_agents):
                q_i = qnow[ag]
                finisher = False
                if q_i > 0.0:
                    for a in cand_lists[ag]:
                        if a == silent_action_idx:
                            continue
                        if per_action_service[ag][a] + 1e-12 >= q_i:
                            finisher = True
                            break
                ratio = (q_i / (per_agent_best[ag] + 1e-12)) if q_i > 0 else 0.0
                order.append((ag, int(finisher), -ratio))
            order.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            assigned = [silent_action_idx] * num_agents
            used_sc = set()
            for (ag, _, _) in order:
                if qnow[ag] <= 0.0:
                    assigned[ag] = silent_action_idx
                    continue
                chosen = None
                for a in cand_lists[ag]:
                    if a == silent_action_idx:
                        continue
                    sc, _ = action_to_rra[a]
                    if sc == -1:
                        continue
                    if sc not in used_sc:
                        chosen = a
                        used_sc.add(sc)
                        break
                if chosen is None:
                    for a in cand_lists[ag]:
                        if a != silent_action_idx:
                            chosen = a
                            used_sc.add(action_to_rra[a][0])
                            break
                if chosen is None:
                    chosen = silent_action_idx
                assigned[ag] = chosen
            
            # Local search improvement
            if local_search_passes > 0:
                for _ in range(local_search_passes):
                    improved = False
                    base_env_ls = copy.deepcopy(run_env)
                    RRA_cur = rra_from_joint_actions(assigned)
                    gr_cur, _ = base_env_ls.step(RRA_cur.copy(), t)
                    best_val = float(np.asarray(gr_cur).reshape(-1)[0])
                    
                    for ag in range(num_agents):
                        if qnow[ag] <= 0.0:
                            continue
                        best_a = assigned[ag]
                        for a in cand_lists[ag]:
                            if a == assigned[ag]:
                                continue
                            trial = assigned.copy()
                            trial[ag] = a
                            tmp_env = copy.deepcopy(run_env)
                            RRA_try = rra_from_joint_actions(trial)
                            gr_try, _ = tmp_env.step(RRA_try.copy(), t)
                            val = float(np.asarray(gr_try).reshape(-1)[0])
                            if val > best_val + 1e-12:
                                best_val = val
                                best_a = a
                                improved = True
                        assigned[ag] = best_a
                    if not improved:
                        break
            
            # Ensure empty agents are silent
            for ag in range(num_agents):
                if qnow[ag] <= 0.0:
                    assigned[ag] = silent_action_idx
            
            # Execute
            RRA = rra_from_joint_actions(assigned)
            global_reward, _ = run_env.step(RRA.copy(), t)
            per_step_rewards.append(float(np.asarray(global_reward).reshape(-1)[0]))
        
        return float(sum(per_step_rewards))
    
    # --- Main loop over locations ---
    all_location_returns: List[List[float]] = []
    
    for loc_idx, location_data in enumerate(test_data_list):
        # CRITICAL: Use train_data, not loaded_veh_data
        env.train_data = location_data
        
        location_returns: List[float] = []
        for trial in range(num_trials_per_location):
            env.new_random_game()
            total_return = run_greedy_episode(env)
            location_returns.append(total_return)
        
        all_location_returns.append(location_returns)
        
        if verbose:
            print(f"Location {loc_idx}: mean={np.mean(location_returns):.3f}, "
                  f"std={np.std(location_returns):.3f}")
    
    # Flatten and compute overall statistics
    all_returns_flat = [r for loc in all_location_returns for r in loc]
    avg_greedy_return = float(np.mean(all_returns_flat))
    
    if verbose:
        _print_baseline_summary("Greedy", all_location_returns, num_trials_per_location,
                                avg_greedy_return, all_returns_flat)
    
    return all_location_returns, avg_greedy_return


def _print_baseline_summary(name: str, all_location_returns: List[List[float]], 
                            num_trials: int, avg_return: float, 
                            all_returns_flat: List[float]):
    """Helper to print baseline summary table."""
    n_locations = len(all_location_returns)
    total_trials = n_locations * num_trials
    
    print(f"\n{'='*70}")
    print(f"{name} Baseline Summary ({n_locations} locations × {num_trials} trials = {total_trials} total):")
    print(f"  Average: {avg_return:.3f}")
    print(f"  Std Dev: {np.std(all_returns_flat):.3f}")
    print(f"{'='*70}")
    
    # Detailed table
    col_width = 12
    header = f"{'Location':<10} "
    for i in range(num_trials):
        header += f"{'T'+str(i+1):<{col_width}}"
    header += f"{'Mean':<{col_width}} {'Std':<{col_width}}"
    print(f"\n{header}")
    print("=" * len(header))
    
    for loc_idx, loc_returns in enumerate(all_location_returns):
        row = f"{loc_idx:<10} "
        for r in loc_returns:
            row += f"{r:<{col_width}.3f}"
        row += f"{np.mean(loc_returns):<{col_width}.3f} {np.std(loc_returns):<{col_width}.3f}"
        print(row)



def compute_exhaustive_baseline(
    env,
    test_data_list: List,
    *,
    num_trials_per_location: int = 1,
    horizon: Optional[int] = None,
    verbose: bool = True,
    print_per_step: bool = False,
) -> Tuple[List[List[float]], float]:
    """
    Compute exhaustive search baseline across all test locations.
    
    WARNING: Exponential complexity in (num_agents * action_dim).
    Only feasible for small problems (e.g., 4 agents with 13 actions each).
    
    Args:
        env: The V2X environment instance
        test_data_list: List of location data (one per test topology)
        num_trials_per_location: Number of rollouts per location (usually 1, deterministic)
        horizon: Episode length (defaults to env.n_step if available, else 50)
        verbose: Whether to print summary results
        print_per_step: Whether to print per-timestep decisions (very verbose)
    
    Returns:
        all_location_returns: List of lists, [n_locations][n_trials] returns
        avg_exhaustive_return: Average return across all trials and locations
    """
    num_agents = env.n_agent
    action_dim = env.n_actions
    horizon = horizon or getattr(env, 'n_step', getattr(env, 'n_step_per_episode', 50))
    
    # --- Build action-to-RRA mapping once ---
    action_to_rra: List[Tuple[int, int]] = []
    silent_action_idx = action_dim - 1  # default
    
    for a in range(action_dim):
        sc, pw = env.map_action_to_rra(a, agent_idx=0)
        sc, pw = int(sc), int(pw)
        action_to_rra.append((sc, pw))
        if (sc, pw) == (-1, -1):
            silent_action_idx = a
    
    def rra_from_joint_actions(joint_actions: List[int]) -> np.ndarray:
        RRA = np.zeros((num_agents, 1, 2), dtype=np.int32)
        for ag in range(num_agents):
            sc, pw = action_to_rra[joint_actions[ag]]
            RRA[ag, 0, 0] = sc
            RRA[ag, 0, 1] = pw
        return RRA
    
    def flatten_q(qarr) -> np.ndarray:
        return np.array(qarr, dtype=float).reshape(-1)
    
    def queues_empty(qarr) -> bool:
        return np.all(flatten_q(qarr) <= 0.0)
    
    def run_exhaustive_episode(base_env) -> float:
        """
        Run one episode with exhaustive search at each timestep.
        """
        run_env = copy.deepcopy(base_env)
        per_step_rewards: List[float] = []
        
        for t in range(horizon):
            qnow = flatten_q(run_env.queue)
            
            if queues_empty(qnow):
                # All queues empty → everyone SILENT
                actions = [silent_action_idx] * num_agents
                RRA = rra_from_joint_actions(actions)
                global_reward, _ = run_env.step(RRA.copy(), t)
                per_step_rewards.append(float(np.asarray(global_reward).reshape(-1)[0]))
                continue
            
            # 1) Determine valid action space for each agent
            valid_actions_per_agent: List[List[int]] = []
            for ag in range(num_agents):
                if qnow[ag] <= 0.0:
                    # Agent has empty queue → must be SILENT
                    valid_actions_per_agent.append([silent_action_idx])
                else:
                    # Agent has data → can choose any action
                    valid_actions_per_agent.append(list(range(action_dim)))
            
            # 2) Generate all valid joint actions (Cartesian product)
            joint_actions_space = [list(ja) for ja in product(*valid_actions_per_agent)]
            
            # 3) Exhaustive search over all valid joint actions
            optimal_joint_action = None
            optimal_reward = -float('inf')
            
            for ja in joint_actions_space:
                tmp_env = copy.deepcopy(run_env)
                RRA = rra_from_joint_actions(ja)
                global_reward, _ = tmp_env.step(RRA.copy(), t)
                reward_value = float(np.asarray(global_reward).reshape(-1)[0])
                
                if reward_value > optimal_reward:
                    optimal_reward = reward_value
                    optimal_joint_action = ja
            
            # 4) Execute the optimal joint action
            RRA = rra_from_joint_actions(optimal_joint_action)
            global_reward, _ = run_env.step(RRA.copy(), t)
            per_step_rewards.append(float(np.asarray(global_reward).reshape(-1)[0]))
            
            if print_per_step:
                print(f"t={t}, queues={qnow}, optimal_action={optimal_joint_action}, "
                      f"reward={optimal_reward:.4f}")
        
        return float(sum(per_step_rewards))
    
    # --- Main loop over locations ---
    all_location_returns: List[List[float]] = []
    
    for loc_idx, location_data in enumerate(test_data_list):
        # CRITICAL: Use train_data, not loaded_veh_data
        env.train_data = location_data
        
        location_returns: List[float] = []
        for trial in range(num_trials_per_location):
            env.new_random_game()
            total_return = run_exhaustive_episode(env)
            location_returns.append(total_return)
        
        all_location_returns.append(location_returns)
        
        if verbose:
            print(f"Location {loc_idx}: Exhaustive return = {np.mean(location_returns):.3f}")
    
    # Flatten and compute overall statistics
    all_returns_flat = [r for loc in all_location_returns for r in loc]
    avg_exhaustive_return = float(np.mean(all_returns_flat))
    
    if verbose:
        _print_baseline_summary("Exhaustive", all_location_returns, num_trials_per_location,
                                avg_exhaustive_return, all_returns_flat)
    
    return all_location_returns, avg_exhaustive_return


def _print_baseline_summary(name: str, all_location_returns: List[List[float]], 
                            num_trials: int, avg_return: float, 
                            all_returns_flat: List[float]):
    """Helper to print baseline summary table."""
    n_locations = len(all_location_returns)
    total_trials = n_locations * num_trials
    
    print(f"\n{'='*70}")
    print(f"{name} Baseline Summary ({n_locations} locations × {num_trials} trials = {total_trials} total):")
    print(f"  Average: {avg_return:.3f}")
    print(f"  Std Dev: {np.std(all_returns_flat):.3f}")
    print(f"{'='*70}")
    
    # Detailed table
    col_width = 12
    header = f"{'Location':<10} "
    for i in range(num_trials):
        header += f"{'T'+str(i+1):<{col_width}}"
    header += f"{'Mean':<{col_width}} {'Std':<{col_width}}"
    print(f"\n{header}")
    print("=" * len(header))
    
    for loc_idx, loc_returns in enumerate(all_location_returns):
        row = f"{loc_idx:<10} "
        for r in loc_returns:
            row += f"{r:<{col_width}.3f}"
        row += f"{np.mean(loc_returns):<{col_width}.3f} {np.std(loc_returns):<{col_width}.3f}"
        print(row)
