# Experiment Results
## Protocol
- Data: TAOBAO-MM user-consistent 1% split, 755,196 train rows and 227,689 test rows
- Hardware: 2 x RTX 4090 24 GB
- DDP: 2 ranks, OMP_NUM_THREADS=1
- Steps: official baselines and warm-up use 376 train steps; architecture adaptations use 100 train steps from the common warm-up checkpoint; all runs use 111 evaluation steps
- Primary metric: GAUC
## Development Results
| Experiment | Seed | Parent checkpoint | GAUC | AUC | LogLoss | Peak GB | Eval steps/s | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-hard | 42 | none | 0.580503 | 0.604105 | 0.387725 | N/M | 6.74 | logs/sim_hard_dev_1pct_retry1.log |
| SIM-soft | 42 | none | 0.580564 | 0.604145 | 0.387717 | N/M | 6.46 | logs/sim_soft_dev_1pct.log |
| MUSE warm-up | 42 | none | 0.584661 | 0.608415 | 0.387076 | N/M | 5.70 | logs/muse_warmup_dev_1pct_retry1.log |
| MUSE continuation | 42 | ckpt/muse_warmup_dev_1pct_{dense,sparse}.ckpt | 0.549213 | 0.571811 | 0.421801 | N/M | 6.21 | logs/muse_continue_dev_1pct.log |
| MUSE short continuation | 42 | ckpt/muse_warmup_dev_1pct_{dense,sparse}.ckpt | 0.583171 | 0.606345 | 0.391455 | N/M | 6.10 | logs/muse_continue_short_dev_1pct.log |
| CP-MUSE | 42 | ckpt/muse_warmup_dev_1pct_{dense,sparse}.ckpt | 0.582923 | 0.606226 | 0.391254 | N/M | 6.10 | logs/cp_muse_dev_1pct.log |

N/M = not measured.

The failed pre-run log `logs/sim_hard_dev_1pct.log` exposed the CLI `use_ddp` override and is excluded from metrics.

The failed warm-up pre-run `logs/muse_warmup_dev_1pct.log` ended after training because `./ckpt` was missing and is excluded; the retry log is valid.

## Decision Log
| Gate | Measured delta | Decision |
| --- | --- | --- |
| Official baselines | N/A | Completed; no gate applies |
| Continuation stability | -0.035448 vs warm-up | Full equal-LR second epoch is consistent with overfitting on the 1% split; revise continuation schedule before architecture comparison. |
| Adaptation schedule | pre-registered before CP/LONGER runs | Use 100 equal continuation steps from the common warm-up checkpoint; full 376-step equal-LR continuation degraded by -0.035448. |
| ETA exploration | CP-MUSE -0.000248 vs 100-step MUSE continuation | Passes the pre-registered -0.0005 gate; retain the ETA-style efficiency ablation. |

CP-MUSE used the same train/evaluation budget as the short continuation control. Its retrieval diagnostics were `RecentOverlap=0.102177` and `RecentEnrichment=1.903319`, showing that the shared SA-TA scorer concentrates selected candidates in the recent window more strongly than its prevalence in the eligible history. This is a retrieval-distribution observation, not evidence of an accuracy gain.
