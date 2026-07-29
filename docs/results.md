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
| MUSE warm-up | 42 | none | 0.584661 | 0.608415 | 0.387076 | N/M | 5.70 | logs/muse_warmup_dev_1pct_retry1.log |
| MUSE continuation | 42 | ckpt/muse_warmup_dev_1pct_{dense,sparse}.ckpt | 0.549213 | 0.571811 | 0.421801 | N/M | 6.21 | logs/muse_continue_dev_1pct.log |

N/M = not measured.

The failed pre-run log `logs/sim_hard_dev_1pct.log` exposed the CLI `use_ddp` override and is excluded from metrics.

The failed warm-up pre-run `logs/muse_warmup_dev_1pct.log` ended after training because `./ckpt` was missing and is excluded; the retry log is valid.

## Decision Log
| Gate | Measured delta | Decision |
| --- | --- | --- |
| Official baselines | N/A | Completed; no gate applies |
| Continuation stability | -0.035448 vs warm-up | Full equal-LR second epoch is consistent with overfitting on the 1% split; revise continuation schedule before architecture comparison. |
