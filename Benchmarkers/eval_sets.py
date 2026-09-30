"""
Evaluation on extra topology sets during training (opt-in via main.py --eval_sets).

At every test point, the trained policy is also run for one episode on each topology of each
extra set. Each set gets its own CSV next to the main result CSV:

    <main result name>_eval-<set name>.csv
    columns: episode, mean, topo_<snapshot_id>, ...   (one row per test point)

The random number generators (Python, NumPy, PyTorch) are saved before and restored after these
evaluations, so enabling them does not change training or the main test results.
"""

import copy
import csv
import os
import random
from contextlib import contextmanager

import numpy as np
import torch as th

from Environment.environment_utility import load_veh_pos


@contextmanager
def preserved_rng():
    """Restore the Python, NumPy and PyTorch RNG states on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    th_state = th.get_rng_state()
    cuda_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        th.set_rng_state(th_state)
        if cuda_state is not None:
            th.cuda.set_rng_state_all(cuda_state)


def with_topologies(params, topologies):
    """Shallow copy of tester params that runs one episode on each topology."""
    q = copy.copy(params)
    q.test_data_list = topologies
    q.num_test_episodes = len(topologies)
    return q


class EvalSets:
    def __init__(self, paths, main_csv_path, n_agent):
        self.sets = []
        for path in paths or []:
            data = load_veh_pos(path)
            topologies = [d for _, d in data.groupby("snapshot_id", sort=False)]
            ids = [int(d["snapshot_id"].iloc[0]) for d in topologies]
            for sid, d in zip(ids, topologies):
                n_tx = int((d["role"] == "Tx").sum())
                if n_tx != n_agent:
                    raise ValueError(
                        f"Eval set {path}: topology {sid} has {n_tx} agents, expected {n_agent}")

            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.splitext(main_csv_path)[0] + f"_eval-{name}.csv"
            f = open(out_path, "w", newline="")
            writer = csv.writer(f)
            writer.writerow(["episode", "mean"] + [f"topo_{sid}" for sid in ids])
            f.flush()
            self.sets.append((name, topologies, f, writer))

    def __bool__(self):
        return bool(self.sets)

    def evaluate(self, episode, run_episodes):
        """
        run_episodes(topologies) -> returns of one episode per topology, in the given order.
        """
        for name, topologies, f, writer in self.sets:
            with preserved_rng():
                returns = np.asarray(run_episodes(topologies), dtype=float).reshape(-1)
            if len(returns) != len(topologies):
                raise RuntimeError(
                    f"Eval set {name}: got {len(returns)} returns for {len(topologies)} topologies")
            writer.writerow([episode, float(np.mean(returns))] + [float(r) for r in returns])
            f.flush()
            print(f"  eval {name}: {np.mean(returns):.2f} ({len(topologies)} topologies)")

    def close(self):
        for _, _, f, _ in self.sets:
            f.close()
