"""
Converts a raw SUMO flow-export CSV (time,id,x,y,angle,speed,lane) into the
processed topology-snapshot format this project's environment actually reads
(snapshot_id,scenario,node_id,role,pair_id,x,y,speed_ms), and appends it to
an existing target file (e.g. Environment/SUMOData/NFIG_k16.csv).

The conversion rule was reverse-engineered by comparing the existing raw
files (NFIG_4ag/8ag/16ag.csv) against their processed counterparts
(NFIG_k4/k8/k16.csv) and verified byte-exact across all 27 combinations
(9 snapshots x 3 agent counts) -- this is not a guess, it's the confirmed
rule those files were actually built with:

- Rows are selected by `time == <the chosen raw timestep>`.
- `id` values of the form `carflow<N>_0.0` / `carflow<N>_1.0` are a V2V
  pair: `_0.0` -> role=Tx, `_1.0` -> role=Rx, pair_id=<N> for both.
- `id` values of the form `carflowV2I_<k>.0` are standalone reference
  points: role=V2I, pair_id=-1.
- `node_id` is a flat counter starting at 0 for this snapshot: all V2V
  pairs first (Tx immediately followed by its Rx, ascending N), then all
  V2I rows last (ascending k).
- `x`, `y`, `speed_ms` are copied directly from the raw row (speed_ms =
  raw `speed`, already in m/s -- no conversion). `angle` and `lane` are
  dropped; they don't appear anywhere in the processed schema.
- `snapshot_id` and `scenario` are supplied by the caller, not derived
  from the raw data.

Usage:
    python convert_raw_topology.py \\
        --raw path/to/new_raw_export.csv \\
        --target Environment/SUMOData/NFIG_k16.csv \\
        --snapshot-id 9 \\
        --scenario nfig

    # If the raw file has more than one distinct `time` value, say which one:
    python convert_raw_topology.py --raw multi_time.csv --target ... \\
        --snapshot-id 10 --scenario nfig --time 108

    # Preview without writing anything:
    python convert_raw_topology.py --raw ... --target ... \\
        --snapshot-id 9 --scenario nfig --dry-run
"""
import argparse
import csv
import re
import sys


FLOW_PATTERN = re.compile(r"^carflow(\d+)_([01])\.0$")
V2I_PATTERN = re.compile(r"^carflowV2I_(\d+)\.0$")


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def convert(raw_rows, snapshot_id, scenario):
    """Apply the confirmed raw -> processed rule to one snapshot's raw rows."""
    flow_rows = {}
    v2i_rows = {}

    for row in raw_rows:
        rid = row["id"]
        m = FLOW_PATTERN.match(rid)
        if m:
            idx, sub = int(m.group(1)), m.group(2)
            flow_rows.setdefault(idx, {})[sub] = row
            continue
        m2 = V2I_PATTERN.match(rid)
        if m2:
            v2i_rows[int(m2.group(1))] = row
            continue
        raise ValueError(
            f"Unrecognized id format: {rid!r} -- expected carflow<N>_0.0/1.0 "
            f"or carflowV2I_<k>.0"
        )

    for idx, sides in flow_rows.items():
        if "0" not in sides or "1" not in sides:
            raise ValueError(f"Flow {idx} is missing one side (need both _0.0 and _1.0)")

    n_pairs = len(flow_rows)
    n_v2i = len(v2i_rows)
    if sorted(flow_rows.keys()) != list(range(n_pairs)):
        raise ValueError(f"Flow indices aren't a contiguous 0..{n_pairs - 1} range: {sorted(flow_rows.keys())}")
    if sorted(v2i_rows.keys()) != list(range(n_v2i)):
        raise ValueError(f"V2I indices aren't a contiguous 0..{n_v2i - 1} range: {sorted(v2i_rows.keys())}")

    out_rows = []
    node_id = 0
    for idx in range(n_pairs):
        tx, rx = flow_rows[idx]["0"], flow_rows[idx]["1"]
        out_rows.append([snapshot_id, scenario, node_id, "Tx", idx, tx["x"], tx["y"], tx["speed"]])
        node_id += 1
        out_rows.append([snapshot_id, scenario, node_id, "Rx", idx, rx["x"], rx["y"], rx["speed"]])
        node_id += 1
    for idx in range(n_v2i):
        row = v2i_rows[idx]
        out_rows.append([snapshot_id, scenario, node_id, "V2I", -1, row["x"], row["y"], row["speed"]])
        node_id += 1

    return out_rows, n_pairs, n_v2i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="Path to the raw SUMO flow-export CSV")
    parser.add_argument("--target", required=True, help="Path to the processed CSV to append to")
    parser.add_argument("--snapshot-id", required=True, type=int, help="snapshot_id to assign this topology")
    parser.add_argument("--scenario", required=True, help='scenario label, e.g. "nfig" or "etsi1" (must match the target file\'s existing convention)')
    parser.add_argument("--time", type=float, default=None, help="Which raw `time` value to use, if the raw file has more than one")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without touching the target file")
    args = parser.parse_args()

    raw_all = load_csv(args.raw)
    if not raw_all:
        print(f"[convert_raw_topology] {args.raw} has no rows.", file=sys.stderr)
        sys.exit(1)

    raw_times = sorted(set(float(r["time"]) for r in raw_all))
    if args.time is not None:
        chosen_time = args.time
        if chosen_time not in raw_times:
            print(f"[convert_raw_topology] --time {chosen_time} not found in raw file. Available: {raw_times}", file=sys.stderr)
            sys.exit(1)
    elif len(raw_times) == 1:
        chosen_time = raw_times[0]
    else:
        print(f"[convert_raw_topology] Raw file has multiple time values {raw_times} -- pass --time to pick one.", file=sys.stderr)
        sys.exit(1)

    raw_rows = [r for r in raw_all if float(r["time"]) == chosen_time]

    try:
        out_rows, n_pairs, n_v2i = convert(raw_rows, args.snapshot_id, args.scenario)
    except ValueError as e:
        print(f"[convert_raw_topology] {e}", file=sys.stderr)
        sys.exit(1)

    # Guard against accidentally duplicating a snapshot_id already in the target
    # (checked in --dry-run too, so a preview correctly warns before you commit).
    existing = load_csv(args.target)
    if any(int(r["snapshot_id"]) == args.snapshot_id for r in existing):
        print(f"[convert_raw_topology] snapshot_id={args.snapshot_id} already exists in {args.target} -- refusing to duplicate.", file=sys.stderr)
        sys.exit(1)

    print(f"Raw time={chosen_time} -> snapshot_id={args.snapshot_id}, scenario={args.scenario!r}")
    print(f"  {n_pairs} V2V pairs ({2 * n_pairs} rows) + {n_v2i} V2I rows = {len(out_rows)} total rows")

    if args.dry_run:
        print("\n--dry-run: not writing anything. Rows that would be appended:")
        for row in out_rows:
            print(row)
        return

    with open(args.target, "a", newline="\n") as f:
        writer = csv.writer(f)
        for row in out_rows:
            writer.writerow(row)

    print(f"Appended {len(out_rows)} rows to {args.target}")


if __name__ == "__main__":
    main()
