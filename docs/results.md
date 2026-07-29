# Experiment Results
## Protocol
- Data: TAOBAO-MM user-consistent 1% split, 755,196 train rows and 227,689 test rows
- Hardware: 2 x RTX 4090 24 GB
- DDP: 2 ranks, OMP_NUM_THREADS=1
- Steps: 376 train, 111 evaluation
- Primary metric: GAUC
## Development Results
| Experiment | Seed | Parent checkpoint | GAUC | AUC | LogLoss | Peak GB | Eval steps/s | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-hard | 42 | none | 0.580503 | 0.604105 | 0.387725 | N/M | 6.74 | logs/sim_hard_dev_1pct_retry1.log |
| SIM-soft | 42 | none | 0.580564 | 0.604145 | 0.387717 | N/M | 6.46 | logs/sim_soft_dev_1pct.log |

N/M = not measured.

The failed pre-run log `logs/sim_hard_dev_1pct.log` exposed the CLI `use_ddp` override and is excluded from metrics.

## Decision Log
| Gate | Measured delta | Decision |
| --- | --- | --- |
| Official baselines | N/A | Completed; no gate applies |
