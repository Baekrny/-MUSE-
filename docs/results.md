# Experiment Results

## Protocol

- Data: TAOBAO-MM user-consistent 1% split, 755,196 train rows and 227,689 test rows

- Hardware: 2 x RTX 4090 24 GB

- DDP: 2 ranks, OMP_NUM_THREADS=1

- Steps: official baselines and warm-up use 376 train steps; architecture adaptations use 100 train steps from the common warm-up checkpoint; all runs use 111 evaluation steps

- Primary metric: GAUC

## Development Results

| Experiment | Seed | Parent checkpoint | GAUC | AUC | LogLoss | Peak GB | Eval steps/s | Log |
|---|---:|---|---:|---:|---:|---:|---:|---|
| SIM-hard | 42 | none | 0.580503 | 0.604105 | 0.387725 | N/M | 6.74 | `logs/sim_hard_dev_1pct_retry1.log` |
| SIM-soft | 42 | none | 0.580564 | 0.604145 | 0.387717 | N/M | 6.46 | `logs/sim_soft_dev_1pct.log` |
| MUSE warm-up | 42 | none | 0.584661 | 0.608415 | 0.387076 | N/M | 5.70 | `logs/muse_warmup_dev_1pct_retry1.log` |
| MUSE continuation | 42 | warm-up checkpoint | 0.549213 | 0.571811 | 0.421801 | N/M | 6.21 | `logs/muse_continue_dev_1pct.log` |
| MUSE short continuation | 42 | warm-up checkpoint | 0.583171 | 0.606345 | 0.391455 | N/M | 6.10 | `logs/muse_continue_short_dev_1pct.log` |
| CP-MUSE | 42 | warm-up checkpoint | 0.582923 | 0.606226 | 0.391254 | N/M | 6.10 | `logs/cp_muse_dev_1pct.log` |
| GroupPool-TA | 42 | warm-up checkpoint | 0.582925 | 0.606261 | 0.391452 | 17.24* | 5.59 | `logs/group_pool_dev_1pct.log` |
| InnerTrans-TA | 42 | warm-up checkpoint | 0.582953 | 0.606369 | 0.391422 | 19.99* | 5.80 | `logs/inner_trans_dev_1pct_retry1.log` |
| GlobalToken-LongerLite | 42 | warm-up checkpoint | 0.583169 | 0.606336 | 0.391455 | 19.69* | 6.24 | `logs/global_token_dev_1pct.log` |
| ETA-CP-MUSE | 42 | warm-up checkpoint | 0.576405 | 0.601245 | 0.393025 | 16.14* | 5.75** | `logs/eta_cp_muse_dev_1pct.log` |

N/M = not measured. Values marked `*` are the highest sampled per-GPU allocations from `nvidia-smi`, not profiler-derived peaks. `5.75**` includes the additional full-exact diagnostic pass and is not an online ETA throughput measurement.

The failed pre-run log `logs/sim_hard_dev_1pct.log` exposed the CLI `use_ddp` override and is excluded from metrics.

The failed warm-up pre-run `logs/muse_warmup_dev_1pct.log` ended after training because `./ckpt` was missing and is excluded; the retry log is valid.

The failed InnerTrans pre-run `logs/inner_trans_dev_1pct.log` hit a CUDA SDPA kernel-grid limit when all 250,000 groups were submitted as one attention batch. The retry chunks flattened groups into batches of at most 50,000 without changing group boundaries, order, parameters, or gradients.

## Decision Log

| Gate                    | Measured delta                                        | Decision                                                                                                                                |
| ----------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Official baselines      | N/A                                                   | Completed; no gate applies                                                                                                              |
| Continuation stability  | -0.035448 vs warm-up                                  | Full equal-LR second epoch is consistent with overfitting on the 1% split; revise continuation schedule before architecture comparison. |
| Adaptation schedule     | pre-registered before CP/LONGER runs                  | Use 100 equal continuation steps from the common warm-up checkpoint; full 376-step equal-LR continuation degraded by -0.035448.         |
| ETA exploration         | CP-MUSE -0.000248 vs 100-step MUSE continuation       | Passes the pre-registered -0.0005 gate; retain the ETA-style efficiency ablation.                                                       |
| GroupPool-TA            | -0.000246 vs 100-step MUSE continuation               | Keep as a runnable LONGER-inspired compression baseline; proceed to the pre-registered InnerTrans comparison.                           |
| GlobalToken exploration | InnerTrans-TA -0.000218 vs 100-step MUSE continuation | Passes the pre-registered -0.001 gate; retain the GlobalToken ablation.                                                                 |
| GlobalToken result      | -0.000002 vs 100-step MUSE continuation               | Numerically matches the control; keep as the most stable LONGER-inspired variant without claiming an accuracy gain.                     |
| ETA result              | -0.006766 vs 100-step MUSE continuation               | Hash Top-200 recall is too low for the final model; keep only as a measured recall-efficiency trade-off.                                |

CP-MUSE used the same train/evaluation budget as the short continuation control. Its retrieval diagnostics were `RecentOverlap=0.102177` and `RecentEnrichment=1.903319`, showing that the shared SA-TA scorer concentrates selected candidates in the recent window more strongly than its prevalence in the eligible history. This is a retrieval-distribution observation, not evidence of an accuracy gain.

GroupPool-TA adds 42,305 dense parameters and compresses the full 1,000-event history into 250 grouped tokens before target attention. It completed on 2 x RTX 4090 without OOM; its near-zero GAUC delta supports using it as the controlled base for the InnerTrans ablation, not claiming an accuracy improvement.

InnerTrans-TA adds 50,849 dense parameters in total and applies one shared local Transformer layer independently inside each four-event group before pooling. It remained within an observed 19.99 GB per GPU and produced the best GAUC among the three adaptations, but its delta against the short MUSE continuation remained negative.

GlobalToken-LongerLite adds 71,137 dense parameters in total. Target, user, CLS, and the most recent 100 merged tokens read all 250 history tokens, while the history representation never reads target or user features. Its GAUC matched the control to six decimal places; because the branch uses a zero-initialized residual scale, this result demonstrates stable integration and directional full-history access, not a proven accuracy contribution from the new branch.

ETA-CP-MUSE uses a fixed 32-bit random-projection hash to shortlist 200 of 1,000 events, then applies the same CP-MUSE SA-TA scorer to select 50. It achieved `Recall@200=0.367484`; the measured retrieval stage fell from `4.259 ms` for full exact CP to `2.590 ms` for hash plus candidate reranking, a 1.64x retrieval-stage speedup (39.2% lower latency), excluding the embedding lookup shared by both paths. The substantial GAUC loss means this is an efficiency ablation, not the selected model or an end-to-end serving speedup claim.

## ETA Inference-Only Shortlist Sweep

To isolate the approximation trade-off from training, all points below load the same MUSE warm-up checkpoint and vary only the 32-bit hash shortlist size. GAUC, AUC, LogLoss, and Recall cover all 111 evaluation steps. CUDA retrieval timing excludes the first two cold-start batches; the cold-start diagnosis and the final sweep use separate immutable logs.

| K | GAUC | AUC | LogLoss | Delta vs K=1000 | Recall@K | ETA ms | Exact ms | Speedup | Latency change |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.576170 | 0.600102 | 0.389545 | -0.008971 | 0.214131 | 2.216 | 4.242 | 1.914x | -47.8% |
| 200 | 0.577476 | 0.602502 | 0.388713 | -0.007665 | 0.361794 | 2.561 | 4.257 | 1.662x | -39.8% |
| 400 | 0.579707 | 0.604111 | 0.387994 | -0.005434 | 0.600948 | 3.648 | 4.242 | 1.163x | -14.0% |
| 600 | 0.581298 | 0.605676 | 0.387562 | -0.003843 | 0.780092 | 4.799 | 4.249 | 0.885x | +12.9% |
| 800 | 0.583547 | 0.607247 | 0.387247 | -0.001594 | 0.906605 | 5.900 | 4.243 | 0.719x | +39.1% |
| 1000 | 0.585141 | 0.608322 | 0.387076 | 0 | 0.999347 | 6.933 | 4.258 | 0.614x | +62.8% |

K=100 is the fastest point, but its GAUC loss is `0.008971`. K=800 is the closest tested approximation to the full-shortlist quality reference, but it still loses `0.001594` GAUC and is slower than exact CP. Therefore, no tested point satisfies the pre-registered `GAUC loss <= 0.0005` constraint while providing a retrieval-stage speedup.

The earlier 100-step architecture adaptations and the ETA operating-point sweep did not pass their respective promotion gates. The staged 300-step GlobalToken follow-up below did pass its gate; second-seed confirmation is now the next accuracy-oriented step before increasing data scale.

## GlobalToken Staged Follow-up

The follow-up uses the same 1% split and warm-up checkpoint, but a matched 300-step low-learning-rate budget. GlobalToken joint starts with `residual_init=0.05` and trains all parameters jointly. GlobalToken staged trains only `longer_lite` for the first 75 steps at the original adaptation rates, then restores the full model and switches to dense/sparse rates `5e-5/5e-4` for the remaining 225 steps.

| Experiment | Train steps | GAUC | AUC | LogLoss | Delta vs MUSE control | Log |
|---|---:|---:|---:|---:|---:|---|
| MUSE low-LR control | 300 | 0.578522 | 0.602126 | 0.394461 | 0 | `logs/muse_low_lr_300_dev_1pct_20260730_run1.log` |
| GlobalToken joint | 300 | 0.578873 | 0.602621 | 0.394256 | +0.000351 | `logs/global_token_joint_300_dev_1pct_20260730_run1.log` |
| GlobalToken staged | 300 | 0.581907 | 0.606298 | 0.391152 | +0.003385 | `logs/global_token_staged_300_dev_1pct_20260730_run1.log` |

The staged run passed the pre-registered `+0.0005` promotion gate and exceeded joint training by `+0.003034` GAUC. The log confirms the branch-only to joint transition at step 75. This is a promising single-seed 1% development result, not yet a paper-level or full-data claim; second-seed confirmation is the next gated experiment.
