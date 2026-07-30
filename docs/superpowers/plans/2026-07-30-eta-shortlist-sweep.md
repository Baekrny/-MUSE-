# ETA Shortlist Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a controlled ETA candidate-size quality/latency curve on the existing TAOBAO-MM 1% test split without retraining or overwriting prior experiments.

**Architecture:** Evaluate the same MUSE warm-up checkpoint through the ETA inference path while varying only `hash_shortlist`. Add a CLI override for the sweep, report the actual K in diagnostics, and store one immutable log per K before summarizing the Pareto frontier.

**Tech Stack:** Python 3.10, PyTorch DDP, pytest, JSON layered configuration, AutoDL 2 x RTX 4090.

---

### Task 1: Final packaging verification

**Files:**
- Verify: `README.md`
- Verify: `README_zh.md`
- Verify: `docs/results.md`
- Verify: `docs/results_zh.md`
- Verify: `scripts/model_smoke.py`
- Verify: `config/final_smoke.json`

- [ ] Compare local and remote copies before synchronization.
- [ ] Synchronize only the intended files that differ.
- [ ] Run the compact core pytest suite and CPU model smoke remotely.
- [ ] Run a two-step GlobalToken DDP smoke with a new log file.

### Task 2: Candidate-size override and truthful diagnostics

**Files:**
- Modify: `main.py`
- Modify: `trainer.py`
- Modify: `tests/test_checkpoint_loading.py`

- [ ] Add a parser test proving `--hash_shortlist 400` is accepted as an integer.
- [ ] Run the test and verify it fails before implementation.
- [ ] Add `--hash_shortlist` to `build_arg_parser`.
- [ ] Change the ETA diagnostic label from fixed `ETARecall@200` to the configured shortlist size.
- [ ] Run the compact test suite.

### Task 3: Evaluation-only sweep configuration

**Files:**
- Create: `config/eta_sweep_eval.json`

- [ ] Configure `job_type=eval`, the common warm-up dense/sparse checkpoints, 111 evaluation steps, ETA diagnostics, and no checkpoint writes.
- [ ] Layer it after the shared 1% continuation and ETA configs so each run differs only by CLI `hash_shortlist` and `exp_name`.
- [ ] Run a two-step K=200 DDP smoke with a new log and verify the dynamic diagnostic label.

### Task 4: Full candidate-size sweep

**Files:**
- Create remotely: `logs/eta_sweep_k100.log`
- Create remotely: `logs/eta_sweep_k200.log`
- Create remotely: `logs/eta_sweep_k400.log`
- Create remotely: `logs/eta_sweep_k600.log`
- Create remotely: `logs/eta_sweep_k800.log`
- Create remotely: `logs/eta_sweep_k1000.log`

- [ ] Evaluate K in `100, 200, 400, 600, 800, 1000` sequentially on two GPUs.
- [ ] Confirm every run reaches 111 evaluation steps and reports GAUC, AUC, LogLoss, Recall@K, ETA total milliseconds, and exact CP milliseconds.
- [ ] Compute speedup and latency reduction from the measured exact reference for each K.
- [ ] Select the fastest point whose GAUC loss versus K=1000 does not exceed `0.0005`; report that no feasible point exists if the constraint is unmet.

### Task 5: Results and decision

**Files:**
- Modify: `docs/results.md`
- Modify: `docs/results_zh.md`
- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] Add the inference-only sweep table and distinguish it from the trained Top-200 ETA experiment.
- [ ] Record the selected operating point and measurement limitations.
- [ ] Decide whether the evidence justifies a 10% MUSE-versus-GlobalToken validation run.
- [ ] Run final verification and commit only task-owned files.
