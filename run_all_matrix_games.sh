#!/bin/bash
# run_all_matrix_games.sh
# Runs IDQL + IPPO on 6 densities × 5 seeds (60 total runs).
#
# Usage:
#   bash run_all_matrix_games.sh            # sequential
#   bash run_all_matrix_games.sh --parallel # parallel (N_PARALLEL jobs at once)

DENSITIES=(35 62 123 167 333 500)
SEEDS=(0 1 2 3 4)
EPISODES=20000
IDQL_EF=0.08
IPPO_EC=0.001
N_PARALLEL=4   # max concurrent jobs in parallel mode

run_sequential() {
    for density in "${DENSITIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[IDQL]  λ=$density  seed=$seed"
            python3 run_matrix_game.py \
                --density $density --seed $seed \
                --episodes $EPISODES --epsi_final $IDQL_EF

            echo "[IPPO]  λ=$density  seed=$seed"
            python3 run_matrix_game_ippo.py \
                --density $density --seed $seed \
                --episodes $EPISODES --entropy_coef $IPPO_EC
        done
    done
}

run_parallel() {
    for density in "${DENSITIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # throttle: wait until fewer than N_PARALLEL jobs running
            while [ "$(jobs -rp | wc -l)" -ge "$N_PARALLEL" ]; do
                sleep 3
            done
            echo "[IDQL]  λ=$density  seed=$seed"
            python3 run_matrix_game.py \
                --density $density --seed $seed \
                --episodes $EPISODES --epsi_final $IDQL_EF &

            while [ "$(jobs -rp | wc -l)" -ge "$N_PARALLEL" ]; do
                sleep 3
            done
            echo "[IPPO]  λ=$density  seed=$seed"
            python3 run_matrix_game_ippo.py \
                --density $density --seed $seed \
                --episodes $EPISODES --entropy_coef $IPPO_EC &
        done
    done
    wait
}

if [[ "$1" == "--parallel" ]]; then
    echo "=== Parallel mode (N_PARALLEL=$N_PARALLEL) ==="
    run_parallel
else
    echo "=== Sequential mode ==="
    run_sequential
fi

echo "=== All 60 runs done ==="
echo "Results in Results/IDQL_MatrixGame/ and Results/IPPO_MatrixGame/"
