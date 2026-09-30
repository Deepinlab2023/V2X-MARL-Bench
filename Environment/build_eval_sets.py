"""
Build the evaluation topology sets for SIG-ML / POSIG (and NFIG) from the SIG-ML datasets.

For each agent count k in {4, 8, 16}, from Environment/SUMOData/SIG_ML_k{k}.csv:

  unseen100_k{k}.csv  100 held-out topologies, removed from SIG-ML / POSIG training
  rep9_k{k}.csv       9 representative topologies, one per (density, BS-distance) cell of unseen100
  seen100_k{k}.csv    100 topologies from the training pool, stratified like unseen100

Stratification follows the original 9 test topologies (Environment/SUMOData/NFIG_k{k}.csv):
3 densities (etsi1 / etsi3 / etsi6) x 3 BS-distance bands (far / mid / near), 11 per cell = 99,
plus 1 extra etsi1-near = 100.

Each scenario cycles through 5 SUMO road windows with snapshot_id:
    window  = (snapshot_id - first_id_of_scenario) % 5
    centres ~ -800, -400, 0, +400, +800 m (BS at x = 0; etsi1 is clipped by the road ends)
    band:     far   mid  near  mid   far
    instant = (snapshot_id - first_id_of_scenario) // 5
A set takes at most one snapshot per (scenario, instant).

rep9 picks, in each cell of unseen100, the topology closest to the cell median of
(|cluster centre x|, cluster span), each feature scaled by its spread within the cell.
Files are ordered etsi1 far, mid, near, etsi3 ..., etsi6 ..., like the original 9 (--loc 0..8).

The metadata of every picked topology is written to eval_sets_meta.csv.

Usage (from the repo root):
    python Environment/build_eval_sets.py
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = "Environment/SUMOData"
OUT_DIR = os.path.join(DATA_DIR, "eval_sets")
AGENT_COUNTS = (4, 8, 16)
SCENARIOS = ["etsi1", "etsi3", "etsi6"]
BANDS = ["far", "mid", "near"]
WINDOW_BAND = {0: "far", 1: "mid", 2: "near", 3: "mid", 4: "far"}
PER_CELL = 11
EXTRA = {("etsi1", "near"): 1}   # 99 + 1 = 100
UNSEEN_SEED = 2026
SEEN_SEED = 2027


def snapshot_table(df):
    g = df.groupby("snapshot_id").agg(
        scenario=("scenario", "first"),
        x_center=("x", "mean"),
        x_span=("x", lambda s: s.max() - s.min()),
    )
    first = g.reset_index().groupby("scenario")["snapshot_id"].min()
    offset = g.index.values - g["scenario"].map(first).values
    g["window"] = offset % 5
    g["instant"] = offset // 5
    g["band"] = g["window"].map(WINDOW_BAND)
    return g


def stratified_sample(table, rng):
    picked = []
    for sc in SCENARIOS:
        used_instants = set()
        for band in BANDS:
            n = PER_CELL + EXTRA.get((sc, band), 0)
            pool = table[(table.scenario == sc) & (table.band == band)]
            pool = pool[~pool.instant.isin(used_instants)]
            # one snapshot per instant within this cell, then n distinct instants
            pool = pool.loc[rng.permutation(pool.index)].drop_duplicates("instant")
            chosen = pool.iloc[:n]
            used_instants.update(chosen.instant)
            picked.append(chosen)
    return pd.concat(picked)


def representative(cell):
    feats = np.column_stack([cell.x_center.abs(), cell.x_span])
    scale = feats.std(axis=0)
    scale[scale == 0] = 1.0
    dist = np.linalg.norm((feats - np.median(feats, axis=0)) / scale, axis=1)
    return cell.iloc[[int(np.argmin(dist))]]


def ordered(sel):
    key = (sel.scenario.map(SCENARIOS.index) * 10 + sel.band.map(BANDS.index))
    return sel.assign(_k=key).sort_values(["_k"], kind="stable").drop(columns="_k")


def write_rows(df, ids, path):
    rows = df[df.snapshot_id.isin(ids)]
    order = {sid: i for i, sid in enumerate(ids)}
    rows = rows.iloc[np.argsort(rows.snapshot_id.map(order).values, kind="stable")]
    rows.to_csv(path, index=False)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta = []
    for k in AGENT_COUNTS:
        df = pd.read_csv(os.path.join(DATA_DIR, f"SIG_ML_k{k}.csv"))
        table = snapshot_table(df)

        unseen = stratified_sample(table, np.random.default_rng(UNSEEN_SEED))
        assert len(unseen) == 100 and unseen.index.is_unique

        rep9 = pd.concat([representative(unseen[(unseen.scenario == sc) & (unseen.band == b)])
                          for sc in SCENARIOS for b in BANDS])

        train_pool = table.drop(index=unseen.index)
        seen = stratified_sample(train_pool, np.random.default_rng(SEEN_SEED))
        assert len(seen) == 100 and not seen.index.isin(unseen.index).any()

        for name, sel in [("rep9", rep9), ("unseen100", unseen), ("seen100", seen)]:
            sel = ordered(sel)
            ids = list(sel.index)
            rows = write_rows(df, ids, os.path.join(OUT_DIR, f"{name}_k{k}.csv"))
            assert rows.snapshot_id.nunique() == len(ids)
            m = sel.reset_index()[["snapshot_id", "scenario", "band", "window", "instant",
                                   "x_center", "x_span"]]
            m.insert(0, "set", name)
            m.insert(0, "k", k)
            meta.append(m)
        print(f"k={k}: unseen100, rep9, seen100 written "
              f"({len(table)} snapshots, {len(train_pool)} left for training)")

    pd.concat(meta).round(1).to_csv(os.path.join(OUT_DIR, "eval_sets_meta.csv"), index=False)


if __name__ == "__main__":
    main()
