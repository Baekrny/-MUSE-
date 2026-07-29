# HA-MUSE Four-Day Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a resume-ready long-sequence recommendation system that compares official SIM/MUSE baselines, TWIN-inspired consistent-relevance retrieval, and LONGER-inspired compressed full-history modeling on the user-consistent 1% TAOBAO-MM development set, with conditional ETA and full-data follow-ups.

**Architecture:** Keep the official MUSE short-interest path and downstream prediction tower intact. Add one shared SA-TA raw relevance API so CP-MUSE uses exactly the same target-history score for full-history Top-K retrieval and ESU attention, then add a separate zero-initialized residual branch that compresses all 1,000 history tokens into a 32-dimensional long-interest vector. All learned variants fork from the same one-epoch MUSE checkpoint and continue for the same number of steps; expensive ETA, GlobalToken, second-seed, and full-data runs are guarded by explicit GAUC thresholds.

**Tech Stack:** Python 3.10, PyTorch 2.6, torch.distributed DDP/Gloo, TAOBAO-MM Open-source-1k, pytest, NumPy, two RTX 4090 GPUs.

---

## Scope And File Map

The existing dual-horizon implementation remains a completed negative ablation. Do not remove it, rewrite its logs, or include its untracked configuration files in unrelated commits.

**Create:**

- `config/sim_hard_dev.json`: official SIM-hard protocol on the fixed 1% development split.
- `config/sim_soft_dev.json`: official SIM-soft protocol on the same split.
- `config/muse_warmup_dev.json`: one-epoch MUSE checkpoint producer.
- `config/muse_continue_dev.json`: equal-step continuation control from the warm-up checkpoint.
- `config/cp_muse_dev.json`: consistent-relevance retrieval config overlay.
- `config/group_pool_dev.json`: GroupPool-TA residual config overlay.
- `config/inner_trans_dev.json`: InnerTrans-TA residual config overlay.
- `config/global_token_dev.json`: conditional GlobalToken-LongerLite config overlay.
- `config/eta_cp_muse_dev.json`: conditional 32-bit ETA-style hash config overlay.
- `model/base_model/longer_lite.py`: GroupPool, InnerTrans, and GlobalToken full-history encoders.
- `utils/hash_retrieval.py`: deterministic random-projection hashing and Hamming shortlist utilities.
- `tests/test_relevance_score.py`: GSU/ESU score identity and masking checks.
- `tests/test_longer_lite.py`: shape, padding, zero-residual, and local-order checks.
- `tests/test_hash_retrieval.py`: deterministic hash and recall checks.
- `docs/results.md`: immutable experiment table, command, checkpoint lineage, and hardware record.

**Modify:**

- `main.py`: training-time checkpoint loading and new method/config routing.
- `trainer.py`: CP-MUSE retrieval, full-history embedding handoff, timing, and retrieval diagnostics.
- `model/muse.py`: optional full-history residual branch and backward-compatible checkpoint state.
- `model/base_model/layers.py`: single raw SA-TA relevance implementation shared by attention and retrieval.
- `scripts/model_smoke.py`: smoke coverage for CP-MUSE and LONGER-Lite variants.
- `README.md`: final reproducible commands, measured results, and bounded resume wording.

No dataset is moved or copied. On the server, keep `/root/autodl-tmp/taobao-mm` in place and retain the existing project symlink or configured relative path.

## Four-Day Critical Path

| Day | Required outcome | Conditional work |
|---|---|---|
| 1 | SIM-hard/SIM-soft runs; warm-up checkpoint; continuation MUSE result | None |
| 2 | Shared relevance API; CP-MUSE test and dev result | ETA only if CP delta is at least `-0.0005` |
| 3 | GroupPool-TA and InnerTrans-TA test and dev results | GlobalToken only if InnerTrans delta is at least `-0.001` |
| 4 | Result audit, second seed when warranted, README/results update | Full data only if first-seed delta is at least `+0.0005` and second-seed delta is non-negative |

Every model comparison uses `batch_size=1000`, `max_train_steps=376`, `max_eval_steps=111`, `keep_top=50`, two GPUs, and the same data order. GAUC is the primary metric; AUC, LogLoss, QPS/retrieval latency, peak memory, and added parameter count are supporting metrics.

### Task 1: Freeze Official Development Baselines

**Files:**
- Create: `config/sim_hard_dev.json`
- Create: `config/sim_soft_dev.json`
- Create: `docs/results.md`

- [ ] **Step 1: Create complete development configurations**

Create `config/sim_hard_dev.json`:

```json
{
  "exp_name": "sim_hard_dev_1pct",
  "job_type": "train",
  "train_data_path": "./taobao-mm-dev/train",
  "test_data_path": "./taobao-mm-dev/test",
  "feature_map_path": "./taobao-mm-dev/feature_map",
  "ckpt_path": "./ckpt",
  "seed": 42,
  "batch_size": 1000,
  "epochs": 1,
  "max_train_steps": 376,
  "max_eval_steps": 111,
  "dense_lr": 0.0002,
  "sparse_lr": 0.002,
  "embedding_dim": 16,
  "method": "sim-hard",
  "keep_top": 50,
  "short_window": 50,
  "use_ddp": true,
  "shuffle": false,
  "shuffle_buffer_size": 1000,
  "use_aux_loss": false,
  "item_id_p90": true,
  "scl_emb_p90": true,
  "feature_map_on_cuda": true,
  "scl_emb_on_cuda": true,
  "save_ckpt": false
}
```

Create `config/sim_soft_dev.json` with the same complete content except for these concrete fields:

```json
{
  "exp_name": "sim_soft_dev_1pct",
  "job_type": "train",
  "train_data_path": "./taobao-mm-dev/train",
  "test_data_path": "./taobao-mm-dev/test",
  "feature_map_path": "./taobao-mm-dev/feature_map",
  "ckpt_path": "./ckpt",
  "seed": 42,
  "batch_size": 1000,
  "epochs": 1,
  "max_train_steps": 376,
  "max_eval_steps": 111,
  "dense_lr": 0.0002,
  "sparse_lr": 0.002,
  "embedding_dim": 16,
  "method": "sim-soft",
  "keep_top": 50,
  "short_window": 50,
  "use_ddp": true,
  "shuffle": false,
  "shuffle_buffer_size": 1000,
  "use_aux_loss": false,
  "item_id_p90": true,
  "scl_emb_p90": true,
  "feature_map_on_cuda": true,
  "scl_emb_on_cuda": true,
  "save_ckpt": false
}
```

- [ ] **Step 2: Validate both JSON files locally**

Run:

```bash
python -m json.tool config/sim_hard_dev.json >/dev/null
python -m json.tool config/sim_soft_dev.json >/dev/null
```

Expected: both commands exit with code 0 and print no error.

- [ ] **Step 3: Run the official baselines on the server**

Run each command separately so every experiment has an independent log:

```bash
cd /root/autodl-tmp/ha-muse
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/sim_hard_dev.json 2>&1 | tee logs/sim_hard_dev_1pct.log
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/sim_soft_dev.json 2>&1 | tee logs/sim_soft_dev_1pct.log
```

Expected: both logs reach `Training finished`, report exactly 376 train steps per rank and 111 evaluation steps per rank, and contain finite GAUC/AUC/loss values. Existing logs must not be reused as output targets.

- [ ] **Step 4: Create the result ledger**

Create `docs/results.md` with this exact schema and enter only measured values:

```markdown
# Experiment Results

## Protocol

- Data: TAOBAO-MM user-consistent 1% split, 755,196 train rows and 227,689 test rows
- Hardware: 2 x RTX 4090 24 GB
- DDP: 2 ranks, `OMP_NUM_THREADS=1`
- Steps: 376 train, 111 evaluation
- Primary metric: GAUC

## Development Results

| Experiment | Seed | Parent checkpoint | GAUC | AUC | LogLoss | Peak GB | Eval QPS | Log |
|---|---:|---|---:|---:|---:|---:|---:|---|

## Decision Log

| Gate | Measured delta | Decision |
|---|---:|---|
```

Expected: no claimed value is copied from a paper, screenshot, or unrelated run.

- [ ] **Step 5: Commit the baseline protocol**

```bash
git add config/sim_hard_dev.json config/sim_soft_dev.json docs/results.md
git commit -m "exp: add fixed SIM development baselines"
```

### Task 2: Add Fair Training-Time Warm Start

**Files:**
- Create: `config/muse_warmup_dev.json`
- Create: `config/muse_continue_dev.json`
- Modify: `main.py`
- Test: `tests/test_checkpoint_loading.py`

- [ ] **Step 1: Write a failing checkpoint-path validation test**

Create `tests/test_checkpoint_loading.py`:

```python
from pathlib import Path

import pytest

from main import validate_warm_start_paths


def test_validate_warm_start_paths_requires_both_files(tmp_path: Path):
    dense = tmp_path / "dense.ckpt"
    dense.touch()
    with pytest.raises(FileNotFoundError, match="sparse warm-start checkpoint"):
        validate_warm_start_paths(
            {
                "warm_start_dense_ckpt": str(dense),
                "warm_start_sparse_ckpt": str(tmp_path / "missing.ckpt"),
            }
        )
```

- [ ] **Step 2: Run the test and verify the missing API failure**

Run:

```bash
pytest tests/test_checkpoint_loading.py -v
```

Expected: collection fails because `validate_warm_start_paths` is not defined.

- [ ] **Step 3: Implement validation and load before DDP wrapping**

Add to `main.py`:

```python
def validate_warm_start_paths(args):
    dense_path = args.get("warm_start_dense_ckpt")
    sparse_path = args.get("warm_start_sparse_ckpt")
    if bool(dense_path) != bool(sparse_path):
        raise ValueError("dense and sparse warm-start checkpoints must be provided together")
    if not dense_path:
        return None
    if not os.path.isfile(dense_path):
        raise FileNotFoundError(f"dense warm-start checkpoint not found: {dense_path}")
    if not os.path.isfile(sparse_path):
        raise FileNotFoundError(f"sparse warm-start checkpoint not found: {sparse_path}")
    return dense_path, sparse_path
```

In `train_and_eval_ddp`, move both models to the selected GPU first, call the validator, load `din_model.load_ckpt(...)` and `embedding_layer.load_ckpt(...)` when paths are present, then construct optimizers from the loaded parameters, and only then wrap both models in DDP. Do not load optimizer state: every continuation branch intentionally starts a fresh optimizer with identical learning rates.

- [ ] **Step 4: Run focused and existing tests**

```bash
pytest tests/test_checkpoint_loading.py tests/test_horizon.py tests/test_horizon_gate.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Create warm-up and continuation configurations**

Create `config/muse_warmup_dev.json`:

```json
{
  "exp_name": "muse_warmup_dev_1pct",
  "job_type": "train",
  "train_data_path": "./taobao-mm-dev/train",
  "test_data_path": "./taobao-mm-dev/test",
  "feature_map_path": "./taobao-mm-dev/feature_map",
  "ckpt_path": "./ckpt",
  "seed": 42,
  "batch_size": 1000,
  "epochs": 1,
  "max_train_steps": 376,
  "max_eval_steps": 111,
  "dense_lr": 0.0002,
  "sparse_lr": 0.002,
  "embedding_dim": 16,
  "method": "muse",
  "keep_top": 50,
  "short_window": 50,
  "long_history_policy": "overlap",
  "use_short_sa_ta": false,
  "use_horizon_gate": false,
  "collect_retrieval_diagnostics": true,
  "use_ddp": true,
  "shuffle": false,
  "shuffle_buffer_size": 1000,
  "use_aux_loss": false,
  "item_id_p90": true,
  "scl_emb_p90": true,
  "feature_map_on_cuda": true,
  "scl_emb_on_cuda": true,
  "save_ckpt": true
}
```

Create `config/muse_continue_dev.json`:

```json
{
  "exp_name": "muse_continue_dev_1pct",
  "job_type": "train",
  "train_data_path": "./taobao-mm-dev/train",
  "test_data_path": "./taobao-mm-dev/test",
  "feature_map_path": "./taobao-mm-dev/feature_map",
  "ckpt_path": "./ckpt",
  "seed": 42,
  "batch_size": 1000,
  "epochs": 1,
  "max_train_steps": 376,
  "max_eval_steps": 111,
  "dense_lr": 0.0002,
  "sparse_lr": 0.002,
  "embedding_dim": 16,
  "method": "muse",
  "keep_top": 50,
  "short_window": 50,
  "long_history_policy": "overlap",
  "use_short_sa_ta": false,
  "use_horizon_gate": false,
  "collect_retrieval_diagnostics": true,
  "use_ddp": true,
  "shuffle": false,
  "shuffle_buffer_size": 1000,
  "use_aux_loss": false,
  "item_id_p90": true,
  "scl_emb_p90": true,
  "feature_map_on_cuda": true,
  "scl_emb_on_cuda": true,
  "save_ckpt": false,
  "warm_start_dense_ckpt": "./ckpt/muse_warmup_dev_1pct_dense.ckpt",
  "warm_start_sparse_ckpt": "./ckpt/muse_warmup_dev_1pct_sparse.ckpt"
}
```

- [ ] **Step 6: Produce and verify the common parent checkpoint**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_warmup_dev.json 2>&1 | tee logs/muse_warmup_dev_1pct.log
test -s ckpt/muse_warmup_dev_1pct_dense.ckpt
test -s ckpt/muse_warmup_dev_1pct_sparse.ckpt
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json 2>&1 | tee logs/muse_continue_dev_1pct.log
```

Expected: both checkpoint files are non-empty; continuation logs explicitly report both loads and finish with finite metrics.

- [ ] **Step 7: Commit warm-start support**

```bash
git add main.py tests/test_checkpoint_loading.py config/muse_warmup_dev.json config/muse_continue_dev.json
git commit -m "feat: add fair warm-start continuation protocol"
```

### Task 3: Unify The SA-TA Raw Relevance Function

**Files:**
- Modify: `model/base_model/layers.py`
- Create: `tests/test_relevance_score.py`

- [ ] **Step 1: Write failing identity and padding tests**

Create `tests/test_relevance_score.py`:

```python
import torch

from model.base_model.layers import MultiHeadAttV2


def make_layer():
    layer = MultiHeadAttV2(4, 4, [4], [4], attn_score_cross=True)
    layer.reset_parameters()
    return layer


def test_forward_uses_exact_raw_relevance_logits():
    torch.manual_seed(7)
    layer = make_layer()
    query = torch.randn(2, 1, 4)
    fact = torch.randn(2, 5, 4)
    mm = torch.randn(2, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)

    captured = {}
    handle = layer.register_forward_pre_hook(lambda _m, args: captured.setdefault("called", True))
    raw = layer.raw_relevance_logits(query, fact, mask=mask, mm_cosine=[mm])
    out = layer(query, fact, mask=mask, mm_cosine=[mm])
    handle.remove()

    assert captured["called"]
    assert raw.shape == (2, layer.heads, 5, 1)
    masked_logits = raw.masked_select(~mask[:, None, :, None].expand_as(raw))
    assert torch.isneginf(masked_logits).all()
    assert out.shape == (2, 4)


def test_calc_attn_score_is_softmax_of_shared_logits():
    torch.manual_seed(11)
    layer = make_layer()
    query = torch.randn(2, 1, 4)
    fact = torch.randn(2, 5, 4)
    mm = torch.randn(2, 5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    raw = layer.raw_relevance_logits(query, fact, mask, [mm])
    expected = torch.softmax(raw, dim=2).squeeze(-1).mean(dim=1)
    actual = layer.calc_attn_score(query, fact, mask, [mm])
    torch.testing.assert_close(actual, expected)
```

- [ ] **Step 2: Run and verify failure**

```bash
pytest tests/test_relevance_score.py -v
```

Expected: failure because `raw_relevance_logits` is absent and `calc_attn_score` does not accept multimodal cosine input.

- [ ] **Step 3: Implement one shared raw-logit API**

Refactor `MultiHeadAttV2` so projection and relevance fusion live in one method with this signature:

```python
def raw_relevance_logits(self, query, fact, mask=None, mm_cosine=None):
    logits = []
    for head in range(self.heads):
        fc_query = self.query_layers[head](query)
        fc_query = fc_query * torch.sigmoid(fc_query)
        fc_fact = self.fact_layers[head](fact)
        fc_fact = fc_fact * torch.sigmoid(fc_fact)
        dot = torch.matmul(fc_fact, fc_query.transpose(-1, -2))
        if self.attn_score_cross and mm_cosine is not None:
            bias = mm_cosine[0] / self.cosine_tau1[0, head]
            if len(mm_cosine) > 1:
                bias = bias + mm_cosine[1] / self.cosine_tau1[1, head]
            bias = bias.reshape(-1, dot.shape[1], 1)
            b1, b2, b3 = self.cosine_tau2[:, head]
            dot = b1 * dot + b2 * bias + b3 * dot * bias
        if mask is not None:
            dot = dot.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
        logits.append(dot)
    return torch.stack(logits, dim=1)
```

Make `forward` call this method once, apply `softmax(dim=2)` per head, and combine each head with its corresponding value projection. Make `calc_attn_score(query, fact, mask=None, mm_cosine=None)` return the head-mean softmax from the same tensor. Preserve the existing `+1e-7` behavior only after softmax if needed for checkpoint-compatible numerics; never add it to masked entries.

- [ ] **Step 4: Prove forward and retrieval share the implementation**

```bash
pytest tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
```

Expected: all tests pass, smoke loss is finite, and tower input dimensions remain unchanged.

- [ ] **Step 5: Commit the scoring refactor**

```bash
git add model/base_model/layers.py tests/test_relevance_score.py
git commit -m "refactor: share SA-TA relevance logits"
```

### Task 4: Implement TWIN-Inspired CP-MUSE Retrieval

**Files:**
- Modify: `main.py`
- Modify: `trainer.py`
- Modify: `scripts/model_smoke.py`
- Create: `config/cp_muse_dev.json`
- Test: `tests/test_relevance_score.py`

- [ ] **Step 1: Add a failing full-history Top-K test**

Append to `tests/test_relevance_score.py`:

```python
from trainer import masked_relevance_topk


def test_masked_relevance_topk_never_returns_padding():
    score = torch.tensor([[0.1, 9.0, 0.8, 0.7]])
    valid = torch.tensor([[True, False, True, True]])
    indices, invalid = masked_relevance_topk(score, valid, keep_top=3)
    assert indices.tolist() == [[2, 3, 0]]
    assert invalid.tolist() == [[False, False, False]]
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_relevance_score.py::test_masked_relevance_topk_never_returns_padding -v
```

Expected: import failure because `masked_relevance_topk` is absent.

- [ ] **Step 3: Implement padding-safe relevance Top-K**

Add to `trainer.py`:

```python
def masked_relevance_topk(score, valid_mask, keep_top):
    masked = score.masked_fill(~valid_mask, float("-inf"))
    available = valid_mask.sum(dim=1)
    k = min(keep_top, score.shape[1])
    indices = torch.topk(masked, k=k, dim=1).indices
    invalid = torch.arange(k, device=score.device)[None, :] >= available[:, None]
    if k < keep_top:
        pad = torch.zeros(score.shape[0], keep_top - k, dtype=torch.long, device=score.device)
        indices = torch.cat([indices, pad], dim=1)
        invalid = torch.cat([invalid, torch.ones_like(pad, dtype=torch.bool)], dim=1)
    return indices, invalid
```

- [ ] **Step 4: Route CP-MUSE through the dense model's shared scorer**

Add `cp-muse` to `create_model` in `main.py`. In `Trainer.apply_general_search`, add a `method == "cp-muse"` branch that:

1. Looks up target ID/category and all 1,000 history ID/category embeddings with `self.sparse_model`.
2. Builds `query = cat(target_item, target_category).unsqueeze(1)` and `fact = cat(history_item, history_category)`.
3. Computes target-history SCL cosine exactly as MUSE already does.
4. Retrieves the unwrapped dense model and calls:

```python
raw = dense_model.uni_att_v2.raw_relevance_logits(
    query,
    fact,
    mask=valid_mask,
    mm_cosine=[mm_cosine],
)
score = raw.squeeze(-1).mean(dim=1)
top_k_indices, invalid_mask = masked_relevance_topk(score, valid_mask, keep_top)
```

5. Gathers IDs, categories, and SCL embeddings using the same indices and zeros every invalid slot.
6. Executes under the existing `@torch.no_grad()` boundary, so discrete Top-K has no gradient; the scorer still learns through ESU because `MUSE_DIN.forward` calls the same `uni_att_v2` parameters with gradient enabled.
7. Records mean raw score, valid candidate count, and retrieval milliseconds in evaluation diagnostics without synchronizing once per sample.

- [ ] **Step 5: Add the CP-MUSE configuration**

Create `config/cp_muse_dev.json` as an explicit overlay loaded after `config/muse_continue_dev.json`:

```json
{
  "exp_name": "cp_muse_dev_1pct",
  "method": "cp-muse",
  "collect_retrieval_diagnostics": true
}
```

- [ ] **Step 6: Extend smoke coverage and run tests**

Make `scripts/model_smoke.py` include a `cp-muse` forward case with 1,000 synthetic history entries, padding in the leftmost 200 positions, and assertions that loss and gradients are finite.

```bash
pytest tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
```

Expected: every test passes; no selected CP-MUSE candidate comes from padding.

- [ ] **Step 7: Run the equal-step CP-MUSE experiment**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/cp_muse_dev.json 2>&1 | tee logs/cp_muse_dev_1pct.log
```

Expected: log confirms the same dense/sparse warm-up checkpoint as continuation MUSE, 376/111 steps, finite metrics, and retrieval diagnostics. Add the measured row and `GAUC(CP-MUSE) - GAUC(continuation MUSE)` to `docs/results.md`.

- [ ] **Step 8: Apply the ETA gate and commit**

If CP-MUSE delta is below `-0.0005`, record `Skip ETA: CP-MUSE delta below gate` in the decision log. Otherwise record `Run ETA` and schedule Task 8 after the required LONGER work.

```bash
git add main.py trainer.py scripts/model_smoke.py config/cp_muse_dev.json tests/test_relevance_score.py docs/results.md
git commit -m "feat: add consistent-relevance CP-MUSE retrieval"
```

### Task 5: Add GroupPool-TA As A Zero-Residual Full-History Branch

**Files:**
- Create: `model/base_model/longer_lite.py`
- Create: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Modify: `trainer.py`
- Create: `config/group_pool_dev.json`

- [ ] **Step 1: Write failing pooling and residual tests**

Create `tests/test_longer_lite.py`:

```python
import torch

from model.base_model.longer_lite import GroupPoolTA


def test_group_pool_is_mask_aware_and_fixed_shape():
    model = GroupPoolTA(input_dim=8, model_dim=4, max_len=8, group_size=4)
    history = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)
    mask = torch.tensor([[False, False, True, True, True, True]])
    merged, merged_mask = model.merge(history, mask)
    assert merged.shape == (1, 2, 4)
    assert merged_mask.tolist() == [[True, True]]
    assert torch.isfinite(merged).all()


def test_zero_residual_scale_preserves_base_interest():
    model = GroupPoolTA(input_dim=8, model_dim=4, max_len=8, group_size=4)
    base = torch.randn(2, 4)
    branch = torch.randn(2, 4)
    torch.testing.assert_close(base + model.residual_scale * branch, base)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_longer_lite.py -v
```

Expected: import failure because `longer_lite.py` does not exist.

- [ ] **Step 3: Implement GroupPool-TA**

Create `model/base_model/longer_lite.py` with `GroupPoolTA(nn.Module)` containing:

```python
self.token_projection = nn.Linear(input_dim, model_dim)
self.position_embedding = nn.Embedding(max_len, model_dim)
self.target_projection = nn.Linear(input_dim, model_dim)
self.residual_scale = nn.Parameter(torch.zeros(()))
self.group_size = group_size
```

`merge(history, mask)` must left-pad sequence length to a multiple of four, project tokens, add absolute positions, reshape to `[B, groups, 4, D]`, perform mask-weighted mean with denominator clamped to one, and return `[B, groups, D]` plus `[B, groups]` validity. `forward(target, history, mask)` must compute scaled dot-product target attention over merged tokens, mask invalid groups before softmax, replace all-invalid rows with zero output, and return `[B, D]`.

- [ ] **Step 4: Wire the optional branch without changing tower width**

In `MUSE_DIN.__init__`, create `GroupPoolTA(input_dim=160, model_dim=2 * D, max_len=1000, group_size=4)` when `longer_variant == "group-pool"`. In `forward`, accept optional `full_seq_embs`; concatenate history item ID, category ID, and 128-D SCL tensors into 160-D tokens, build the target token the same way, compute the branch, and apply:

```python
uni_seq_att_out_v2 = (
    uni_seq_att_out_v2
    + self.longer_lite.residual_scale * full_history_interest
)
```

In `Trainer.forward_step`, preserve full-history item/category/SCL embeddings before GSU only when `longer_variant` is enabled and pass them as `full_seq_embs`. This keeps ordinary MUSE memory use unchanged.

- [ ] **Step 5: Make checkpoint loading backward compatible**

Add `longer_lite` to dense checkpoint state only when present. During warm-start loading, load it only if both the model and checkpoint contain the key; otherwise retain its initialized weights and log that the optional branch starts with zero residual scale. Existing MUSE checkpoints must load without error.

- [ ] **Step 6: Run focused tests and smoke**

```bash
pytest tests/test_longer_lite.py tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
```

Expected: all tests pass; a newly enabled GroupPool branch produces exactly the warm-start MUSE output before its first optimization step within numerical tolerance.

- [ ] **Step 7: Create and run the GroupPool configuration**

Create `config/group_pool_dev.json` as an overlay loaded after `config/muse_continue_dev.json`:

```json
{
  "exp_name": "group_pool_dev_1pct",
  "method": "muse",
  "longer_variant": "group-pool",
  "longer_group_size": 4,
  "longer_model_dim": 32
}
```

Run:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/group_pool_dev.json 2>&1 | tee logs/group_pool_dev_1pct.log
```

Expected: 1,000 raw events become 250 merged tokens; training completes without out-of-memory; result ledger includes metrics, added parameters, peak memory, and QPS.

- [ ] **Step 8: Commit GroupPool-TA**

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py trainer.py config/group_pool_dev.json docs/results.md
git commit -m "feat: add GroupPool full-history residual"
```

### Task 6: Add InnerTrans-TA

**Files:**
- Modify: `model/base_model/longer_lite.py`
- Modify: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Create: `config/inner_trans_dev.json`

- [ ] **Step 1: Write a failing local-order and shape test**

Append to `tests/test_longer_lite.py`:

```python
from model.base_model.longer_lite import InnerTransTA


def test_inner_transformer_preserves_group_count_and_masks_padding():
    torch.manual_seed(3)
    model = InnerTransTA(input_dim=8, model_dim=4, max_len=8, group_size=4, num_heads=1)
    history = torch.randn(2, 7, 8)
    mask = torch.tensor([[False, True, True, True, True, True, True], [False] * 7])
    merged, merged_mask = model.merge(history, mask)
    assert merged.shape == (2, 2, 4)
    assert merged_mask.tolist() == [[True, True], [False, False]]
    assert torch.equal(merged[1], torch.zeros_like(merged[1]))
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_longer_lite.py::test_inner_transformer_preserves_group_count_and_masks_padding -v
```

Expected: import failure because `InnerTransTA` is absent.

- [ ] **Step 3: Implement one local Transformer layer per shared group**

Subclass `GroupPoolTA`. Add one `nn.TransformerEncoderLayer(d_model=model_dim, nhead=1, dim_feedforward=2 * model_dim, dropout=0.0, batch_first=True, norm_first=True)`. In `merge`, flatten groups to `[B * groups, 4, D]`, run the same layer over every group with `src_key_padding_mask`, mask invalid outputs to zero, then perform mask-aware mean. Never run attention across group boundaries.

- [ ] **Step 4: Route and test the variant**

In `MUSE_DIN`, construct `InnerTransTA(input_dim=160, model_dim=2 * D, max_len=1000, group_size=4, num_heads=1)` when `longer_variant == "inner-trans"`. Pass the same 160-D target/history tensors and validity mask used by GroupPool, then add `self.longer_lite.residual_scale * full_history_interest` to `uni_seq_att_out_v2`. Save `longer_lite.state_dict()` only when the branch exists; when loading a plain MUSE checkpoint, retain initialized branch parameters and the zero residual scale.

```bash
pytest tests/test_longer_lite.py tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
```

Expected: all tests pass and all-padding history returns a finite zero branch.

- [ ] **Step 5: Create and run the InnerTrans configuration**

Create `config/inner_trans_dev.json` as a complete variant overlay:

```json
{
  "exp_name": "inner_trans_dev_1pct",
  "method": "muse",
  "longer_variant": "inner-trans",
  "longer_group_size": 4,
  "longer_model_dim": 32,
  "longer_num_heads": 1
}
```

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/inner_trans_dev.json 2>&1 | tee logs/inner_trans_dev_1pct.log
```

Expected: finite metrics and a recorded GAUC delta against the same-seed continuation MUSE.

- [ ] **Step 6: Apply the GlobalToken gate and commit**

If InnerTrans delta is below `-0.001`, record `Skip GlobalToken` and continue to Task 9. Otherwise record `Run GlobalToken` and execute Task 7.

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py config/inner_trans_dev.json docs/results.md
git commit -m "feat: add InnerTrans local sequence modeling"
```

### Task 7: Conditionally Add GlobalToken-LongerLite

Execute this task only when the Task 6 gate passes.

**Files:**
- Modify: `model/base_model/longer_lite.py`
- Modify: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Create: `config/global_token_dev.json`

- [ ] **Step 1: Write a failing directionality test**

Append to `tests/test_longer_lite.py`:

```python
from model.base_model.longer_lite import GlobalTokenLongerLite


def test_global_tokens_read_history_in_one_direction():
    torch.manual_seed(5)
    model = GlobalTokenLongerLite(
        input_dim=160,
        user_dim=88,
        model_dim=32,
        max_len=1000,
        group_size=4,
        recent_queries=100,
        num_heads=1,
    )
    target = torch.randn(2, 160)
    user = torch.randn(2, 88)
    history = torch.randn(2, 1000, 160)
    mask = torch.ones(2, 1000, dtype=torch.bool)
    mask[1, :200] = False
    seen = {}

    def capture_lengths(_module, inputs):
        seen["query"] = inputs[0].shape[1]
        seen["key"] = inputs[1].shape[1]
        seen["value"] = inputs[2].shape[1]

    handle = model.cross_attention.register_forward_pre_hook(capture_lengths)
    output = model(target, user, history, mask)
    handle.remove()

    assert seen == {"query": 103, "key": 250, "value": 250}
    assert output.shape == (2, 32)
    assert torch.isfinite(output).all()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_longer_lite.py -k global_token -v
```

Expected: failure because `GlobalTokenLongerLite` is absent.

- [ ] **Step 3: Implement one-way global reading**

Build on InnerTrans merged tokens. Define learnable CLS, target/user projections, `nn.MultiheadAttention(32, 1, batch_first=True)` for cross-attention, and one `TransformerEncoderLayer(32, 1, 64, dropout=0.0, batch_first=True, norm_first=True)` over the 103 query outputs. Queries read all 250 history keys/values; history tokens are never updated using target or user tokens. Return the transformed CLS position.

- [ ] **Step 4: Route, test, and run**

Create `config/global_token_dev.json` as a complete variant overlay:

```json
{
  "exp_name": "global_token_dev_1pct",
  "method": "muse",
  "longer_variant": "global-token",
  "longer_group_size": 4,
  "longer_model_dim": 32,
  "longer_num_heads": 1,
  "longer_recent_queries": 100
}
```

Run:

```bash
pytest tests/test_longer_lite.py -v
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/global_token_dev.json 2>&1 | tee logs/global_token_dev_1pct.log
```

Expected: tests pass; run reports measured parameters, peak memory, QPS, and metrics.

- [ ] **Step 5: Commit the conditional exploration**

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py config/global_token_dev.json docs/results.md
git commit -m "feat: explore global-token long-history encoding"
```

### Task 8: Conditionally Add ETA-Style Hash Shortlisting

Execute this task only when Task 4 recorded CP-MUSE delta at or above `-0.0005`.

**Files:**
- Create: `utils/hash_retrieval.py`
- Create: `tests/test_hash_retrieval.py`
- Modify: `trainer.py`
- Create: `config/eta_cp_muse_dev.json`

- [ ] **Step 1: Write deterministic hash and recall tests**

Create `tests/test_hash_retrieval.py`:

```python
import torch

from utils.hash_retrieval import RandomProjectionHash, topk_recall


def test_random_projection_hash_is_seed_deterministic():
    x = torch.randn(2, 5, 8)
    first = RandomProjectionHash(8, 32, seed=42)(x)
    second = RandomProjectionHash(8, 32, seed=42)(x)
    assert torch.equal(first, second)
    assert first.dtype == torch.bool
    assert first.shape == (2, 5, 32)


def test_topk_recall_excludes_padding():
    exact = torch.tensor([[2, 3, 0]])
    approx = torch.tensor([[2, 1, 3, 0]])
    exact_invalid = torch.tensor([[False, False, True]])
    assert topk_recall(exact, approx, exact_invalid).item() == 1.0
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_hash_retrieval.py -v
```

Expected: import failure because `hash_retrieval.py` is absent.

- [ ] **Step 3: Implement deterministic Hamming shortlisting**

`RandomProjectionHash` must create its `[input_dim, bits]` Gaussian projection using a CPU `torch.Generator().manual_seed(seed)`, register it as a buffer, and return `(x @ projection) >= 0`. Add `hamming_topk(query_code, history_code, valid_mask, keep_top=200)` using boolean XOR and masked Hamming distance. Add `topk_recall` that ignores invalid exact slots and returns the mean fraction of exact IDs present in the approximate shortlist.

- [ ] **Step 4: Add two-stage ETA retrieval**

In the ETA branch of `apply_general_search`, concatenate task-trained ID and category embeddings, hash target and history to 32 bits, select Hamming Top-200, gather these candidates, compute the exact shared CP-MUSE raw relevance `r_i`, then select Top-50. Record exact CP Top-50 recall, Hamming time, exact rerank time, total retrieval time, and peak memory. Do not label it an acceleration unless measured end-to-end retrieval time is lower than exact CP-MUSE.

- [ ] **Step 5: Test and run the conditional experiment**

Create `config/eta_cp_muse_dev.json` as an explicit overlay loaded after the continuation config:

```json
{
  "exp_name": "eta_cp_muse_dev_1pct",
  "method": "eta-cp-muse",
  "hash_bits": 32,
  "hash_shortlist": 200,
  "hash_seed": 42
}
```

```bash
pytest tests/test_hash_retrieval.py tests/test_relevance_score.py -v
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/eta_cp_muse_dev.json 2>&1 | tee logs/eta_cp_muse_dev_1pct.log
```

Expected: deterministic tests pass; experiment records Top-50 recall and measured efficiency alongside ranking metrics.

- [ ] **Step 6: Commit ETA as a bounded ablation**

```bash
git add utils/hash_retrieval.py tests/test_hash_retrieval.py trainer.py main.py config/eta_cp_muse_dev.json docs/results.md
git commit -m "feat: add ETA-style hash retrieval ablation"
```

### Task 9: Verify, Select, And Document The Final Project

**Files:**
- Modify: `scripts/model_smoke.py`
- Modify: `docs/results.md`
- Modify: `README.md`
- Create when gate passes: one explicitly named seed config matching the measured winner, such as `config/cp_muse_seed2026_dev.json`, `config/group_pool_seed2026_dev.json`, `config/inner_trans_seed2026_dev.json`, or `config/global_token_seed2026_dev.json`.
- Create when both gates pass: the corresponding explicitly named full-data config, such as `config/cp_muse_full.json`, `config/group_pool_full.json`, `config/inner_trans_full.json`, or `config/global_token_full.json`.
- Create: `config/final_smoke.json`

- [ ] **Step 1: Run the complete lightweight verification suite**

```bash
pytest tests -v
python scripts/model_smoke.py
```

Expected: all tests pass; every configured model variant has finite forward/backward output; no checkpoint, dataset, log, or existing result is overwritten.

- [ ] **Step 2: Run a two-GPU smoke before any additional expensive experiment**

Create `config/final_smoke.json` as a later config overlay:

```json
{
  "exp_name": "final_two_gpu_smoke",
  "max_train_steps": 2,
  "max_eval_steps": 2,
  "save_ckpt": false
}
```

Run exactly one of the following commands, selecting the config with the highest measured GAUC that has completed successfully:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/cp_muse_dev.json config/final_smoke.json
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/group_pool_dev.json config/final_smoke.json
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/inner_trans_dev.json config/final_smoke.json
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/global_token_dev.json config/final_smoke.json
```

Expected: only the selected command is run; both ranks finish, take equal steps, and exit cleanly. The second JSON overrides only the experiment name and step caps because `main.py` merges config files from left to right.

- [ ] **Step 3: Apply the second-seed gate**

Calculate each eligible variant's GAUC delta against `muse_continue_dev_1pct`. If the best delta is below `+0.0005`, stop development experiments and document the best measured trade-off without claiming an improvement. If the delta is at least `+0.0005`, create a seed-2026 continuation control and winner config, both forked from a seed-2026 warm-up checkpoint, then run both with identical steps.

Expected: `docs/results.md` contains first-seed and second-seed deltas against their matching continuation controls, never against the initial one-epoch MUSE result.

- [ ] **Step 4: Apply the full-data gate**

Proceed only if the second-seed winner delta is non-negative. Create the full-data config by changing paths to `./taobao-mm/train`, `./taobao-mm/test`, and `./taobao-mm/feature_map`, removing `max_train_steps` and `max_eval_steps`, retaining `batch_size=1000`, and using a new experiment/checkpoint/log name.

Before launch, verify free disk space is at least 100 GB beyond the existing 139 GB dataset and checkpoints, system RAM is at least 128 GB, and both 24 GB GPUs are visible. Use `item_id_p90=true`; if GPU memory is insufficient, set `scl_emb_on_cuda=false` before reducing batch size so comparison semantics remain stable.

- [ ] **Step 5: Update the README with measured, bounded claims**

Add sections for:

```markdown
## Project Contributions

- Reproduced official SIM-hard, SIM-soft, and MUSE under one fixed TAOBAO-MM protocol.
- Implemented CP-MUSE, where GSU retrieval and ESU attention share the exact SA-TA relevance function.
- Compared retrieval-based modeling with LONGER-inspired GroupPool/InnerTrans full-history compression.
- Evaluated ETA-style hashing only as an approximate-retrieval trade-off when its prerequisite gate passed.

## Reproduction

Commands, checkpoint lineage, hardware, step counts, and links to `docs/results.md`.

## Limitations

The public sequence length is 1,000 and has no event timestamps. This project does not claim a full reproduction of TWIN, ETA, or LONGER, industrial caching, 100K-length support, or gains not present in the measured table.
```

Use the actual result to write one resume bullet. If GAUC improves, state the measured delta and protocol. If it does not, state that the project built and benchmarked consistent retrieval and compressed full-history alternatives, and report the best efficiency/accuracy trade-off without positive-gain wording.

- [ ] **Step 6: Audit experiment lineage and repository changes**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

Expected: no whitespace errors; every result row names its log and parent checkpoint; unrelated untracked historical configs/scripts remain untouched.

- [ ] **Step 7: Commit final documentation**

```bash
git add README.md docs/results.md scripts/model_smoke.py config
git commit -m "docs: report long-sequence recommendation experiments"
```

Before running this command, inspect `git status --short` and stage individual new winner configs instead of the whole `config` directory if unrelated untracked config files are still present.

## Final Acceptance Checklist

- [ ] SIM-hard, SIM-soft, warm-up MUSE, and continuation MUSE use the same 1% split and capped DDP steps.
- [ ] CP-MUSE GSU and ESU call one raw relevance implementation containing ID/category interaction, SCL bias, and the learned multiplicative cross term.
- [ ] Padding cannot enter Top-K, group pooling, InnerTrans attention, GlobalToken attention, or ETA recall denominators.
- [ ] GroupPool and InnerTrans consume all 1,000 events and produce 250 merged tokens with a 32-D zero-initialized residual.
- [ ] Every learned variant starts from the same matching warm-up checkpoint and receives equal continuation steps.
- [ ] Conditional tasks are executed only after their documented thresholds pass.
- [ ] Results include GAUC, AUC, LogLoss, peak memory, QPS/latency, added parameters, log path, seed, and checkpoint lineage.
- [ ] All tests and the two-GPU smoke pass before a second-seed or full-data launch.
- [ ] README language distinguishes inspiration/ablation from complete ETA, TWIN, LONGER, or industrial-system reproduction.
- [ ] No dataset, checkpoint, log, or pre-existing user file is deleted, moved, cleaned, or overwritten.
