# Experimental configs

Sparse overrides applied on top of the task preset with `--config`. Not loaded by default.

| File | For | What | Status |
|------|-----|------|--------|
| `ppo_SIG_SL_stable.json` | `ippo`, `mappo` on SIG single-location | Raw-return critic (`popart: false`), batch 64, KL early stop (`target_kl` 0.02), KL-adaptive actor LR, Adam eps 1e-5, grad clip 0.5 | IPPO: no late-training collapse over 30000 episodes (loc1, 16 agents, seed 9 only, fast fading off). MAPPO: not validated. |
| `idql_gnn.json` | `idql`, `hys` on SIG (SL or ML) | Q-network = one-round GNN over the SIG global state (nodes: agents' links; edges: interference both ways; message/hidden dim 64) instead of the MLP | Not validated yet. |

```bash
python main.py --env SIG --algo ippo --loc 1 --config Configuration/experimental/ppo_SIG_SL_stable.json
```
