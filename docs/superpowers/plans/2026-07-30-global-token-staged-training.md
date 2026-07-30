# GlobalToken Staged Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled GlobalToken adaptation schedule that avoids zero-residual gradient suppression and compares staged training against matched MUSE and joint-training controls.

**Architecture:** Make the longer-history residual initialization configurable. During the first 75 steps of the staged arm, freeze the MUSE backbone and sparse embedding model while training only `longer_lite`; then restore the original trainability flags and switch both optimizers to lower joint-learning rates for the remaining 225 steps.

**Tech Stack:** Python 3.10, PyTorch, PyTorch DDP, pytest, layered JSON configs.

---

### Task 1: Configurable residual initialization

**Files:**
- Modify: `model/base_model/longer_lite.py`
- Modify: `model/muse.py`
- Modify: `tests/test_longer_lite.py`

- [ ] Add a failing test that constructs `GroupPoolTA(..., residual_init=0.05)` and checks the parameter value.
- [ ] Add a finite float `residual_init` constructor argument, store it, and initialize `residual_scale` with that value.
- [ ] Pass `longer_residual_init` from MUSE config to all LongerLite variants.
- [ ] Run `tests/test_longer_lite.py`.

### Task 2: Staged trainability and learning-rate transition

**Files:**
- Modify: `trainer.py`
- Create: `tests/test_staged_training.py`

- [ ] Test that stage 1 freezes every dense parameter outside `longer_lite`, freezes sparse parameters, and leaves longer parameters trainable.
- [ ] Test that the transition restores original `requires_grad` flags exactly once and changes dense/sparse optimizer learning rates.
- [ ] Implement a small `StagedLongerController` with `start()` and `maybe_transition(step)` methods.
- [ ] Activate the controller only when `staged_longer_warmup_steps > 0`, before the first training forward pass and before each subsequent step.
- [ ] Log stage boundaries only on rank 0.

### Task 3: Matched experiment configs

**Files:**
- Create: `config/muse_low_lr_300_dev.json`
- Create: `config/global_token_joint_300_dev.json`
- Create: `config/global_token_staged_300_dev.json`
- Create: `config/global_token_staged_smoke.json`

- [ ] Configure all three formal arms for 300 training steps and 111 evaluation steps from the common warm-up checkpoint.
- [ ] Use `dense_lr=0.00005` and `sparse_lr=0.0005` for MUSE and joint controls.
- [ ] Configure staged training with 75 branch-only steps at `0.0002`, then joint rates `0.00005` and `0.0005`, plus residual initialization `0.05`.
- [ ] Configure the CPU/DDP smoke to transition after one step and stop after two steps.

### Task 4: Verification and GPU handoff

**Files:**
- Modify: `scripts/model_smoke.py`
- Modify: `README_zh.md`

- [ ] Add a staged-controller smoke that verifies both stages and finite backward gradients without TAOBAO-MM.
- [ ] Run the compact core test suite and CPU smoke in the no-GPU server environment.
- [ ] Synchronize only changed files after comparing remote copies.
- [ ] Prepare three two-GPU commands with new immutable log names; do not start them until the user enables GPUs.

## Execution Checkpoint

Resumed and completed on 2026-07-30 after the user enabled the two GPUs.

- Tasks 1-3 are implemented.
- Targeted tests passed: `7 passed`.
- Compact core suite passed: `35 passed in 50.50s`.
- CPU model smoke passed with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; all three LongerLite variants produced finite forward/backward values and the staged controller transitioned from branch-only to joint mode.
- GPU DDP smoke passed and all three 300-step experiments completed.
- MUSE control GAUC: `0.578522`; GlobalToken joint: `0.578873`; GlobalToken staged: `0.581907`.
- Staged delta versus control: `+0.003385`, exceeding the `+0.0005` gate; second-seed confirmation is pending.
- Chinese execution record: `docs/global_token_staged_status_zh.md`.
