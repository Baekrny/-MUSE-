# HA-MUSE 四天实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在四天内基于阿里官方 MUSE 和 TAOBAO-MM，完成一个可写入推荐算法实习简历的超长行为序列建模项目，严格比较 SIM-hard、SIM-soft、MUSE、TWIN-inspired CP-MUSE 与 LONGER-Lite，并按实验门槛决定是否追加 ETA、第二随机种子和全量训练。

**Architecture:** 保持官方 MUSE 的短期兴趣通路和下游预测塔不变。CP-MUSE 抽取唯一的 SA-TA 原始相关性函数，让 GSU 的全历史 Top-K 检索和 ESU 的候选注意力使用完全相同的打分；LONGER-Lite 作为独立的全历史残差分支，把 1,000 条行为压缩成 250 个 token，并通过初始化为 0 的残差系数接入 MUSE。所有含新增参数的模型都从同一个 MUSE warm-up checkpoint 分叉并等步数续训。

**Tech Stack:** Python 3.10、PyTorch 2.6、PyTorch DDP/Gloo、TAOBAO-MM Open-source-1k、pytest、NumPy、2 x RTX 4090 24GB。

---

## 一、边界与理论依据

### 1. 项目解决的问题

TAOBAO-MM 为每个样本提供最长 1,000 条有序 item/category 行为和 SCL 多模态表示。直接对 1,000 个 token 做复杂注意力成本较高，而只保留最近行为又会损失长期兴趣，因此项目比较两条互补路线：

1. **搜索式路线：** 从完整历史中检索与当前目标最相关的 Top-50，再做精细兴趣建模。
2. **压缩式路线：** 保留全历史覆盖，将相邻行为压缩后进行轻量建模。

### 2. 为什么采用这些改进

- **SIM** 提供 GSU 粗筛加 ESU 精排的两阶段范式，是官方可运行基线。
- **MUSE** 证明多模态余弦适合轻量 GSU，并在 ESU 用 SA-TA 融合 ID 相关性和多模态相关性。
- **TWIN** 的关键不是简单共享 embedding，而是 GSU 与 ESU 使用相同的目标-行为相关性度量。因此 CP-MUSE 必须让两阶段调用同一个 raw score API。
- **ETA** 用随机投影哈希近似目标注意力，但 1K 序列在 GPU 上未必真正加速，所以只把它作为有条件的效率消融，必须报告召回率和实测耗时。
- **LONGER** 的 Token Merge、InnerTrans、Global Token 可以降低长序列建模开销。公开数据没有行为时间戳，因此 LONGER-Lite 只使用绝对位置和新旧顺序，不引入时间差特征。

### 3. 不能越界的表述

可以表述为“借鉴 TWIN/ETA/LONGER 核心思想并在 MUSE 上实现和评估”；不能表述为完整复现 TWIN、ETA 或 LONGER，也不能宣称支持 100K 序列、工业级缓存或尚未测得的 GAUC 提升。

## 二、统一实验协议

- 开发集：用户一致的 1% 子集，训练 755,196 行，测试 227,689 行。
- 硬件：2 x RTX 4090 24GB，`OMP_NUM_THREADS=1`。
- 公平协议：`seed=42`、`batch_size=1000`、`keep_top=50`、`max_train_steps=376`、`max_eval_steps=111`。
- 主指标：GAUC；辅助指标：AUC、LogLoss、QPS/检索延迟、峰值显存、新增参数量。
- warm-start：先训练一轮 MUSE，再从同一 dense/sparse checkpoint 分叉续训一轮。
- 日志、checkpoint 和结果文件使用新名称，不覆盖已有实验。

决策门槛：

```text
GlobalToken：InnerTrans 相对 continuation MUSE 的 GAUC delta >= -0.001
ETA：CP-MUSE 相对 continuation MUSE 的 GAUC delta >= -0.0005
第二 seed：开发集最佳 delta >= +0.0005
全量数据：第二 seed 对应 delta >= 0
```

## 三、文件规划

**新增：**

```text
config/sim_hard_dev.json
config/sim_soft_dev.json
config/muse_warmup_dev.json
config/muse_continue_dev.json
config/cp_muse_dev.json
config/group_pool_dev.json
config/inner_trans_dev.json
config/global_token_dev.json
config/eta_cp_muse_dev.json
config/final_smoke.json
model/base_model/longer_lite.py
utils/hash_retrieval.py
tests/test_checkpoint_loading.py
tests/test_relevance_score.py
tests/test_longer_lite.py
tests/test_hash_retrieval.py
docs/results.md
```

**修改：**

```text
main.py
trainer.py
model/muse.py
model/base_model/layers.py
scripts/model_smoke.py
README.md
```

不移动或复制 `/root/autodl-tmp/taobao-mm`，不清理任何已有日志、checkpoint、配置或未提交文件。

## 四、四天执行顺序

| 天数 | 必须完成 | 满足门槛后才做 |
|---|---|---|
| 第 1 天 | SIM-hard、SIM-soft；MUSE warm-up；continuation MUSE | 无 |
| 第 2 天 | 统一 raw relevance API；CP-MUSE | CP delta 达标后保留 ETA 任务 |
| 第 3 天 | GroupPool-TA；InnerTrans-TA | InnerTrans delta 达标后做 GlobalToken |
| 第 4 天 | 总体验证、结果审计、README 和简历表述 | 第二 seed、全量数据 |

### Task 1：冻结官方基线

**文件：**
- Create: `config/sim_hard_dev.json`
- Create: `config/sim_soft_dev.json`
- Create: `docs/results.md`

- [ ] **Step 1：建立完整的 SIM-hard 与 SIM-soft 配置**

两份配置都使用统一协议。唯一差异如下：

```json
{
  "exp_name": "sim_hard_dev_1pct",
  "method": "sim-hard"
}
```

```json
{
  "exp_name": "sim_soft_dev_1pct",
  "method": "sim-soft"
}
```

其余字段逐项写入两份 JSON：开发集三个路径、`seed=42`、`batch_size=1000`、`epochs=1`、376/111 步、`dense_lr=2e-4`、`sparse_lr=2e-3`、`embedding_dim=16`、`keep_top=50`、`use_ddp=true`、`item_id_p90=true`、`scl_emb_p90=true`、两种映射均放 GPU、`save_ckpt=false`。

- [ ] **Step 2：校验 JSON**

```bash
python -m json.tool config/sim_hard_dev.json >/dev/null
python -m json.tool config/sim_soft_dev.json >/dev/null
```

Expected：两条命令退出码均为 0。

- [ ] **Step 3：远程运行官方基线**

```bash
cd /root/autodl-tmp/ha-muse
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/sim_hard_dev.json 2>&1 | tee logs/sim_hard_dev_1pct.log
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/sim_soft_dev.json 2>&1 | tee logs/sim_soft_dev_1pct.log
```

Expected：两次训练均完成 376/111 步并输出有限的 GAUC、AUC 和 loss。

- [ ] **Step 4：建立结果台账**

`docs/results.md` 固定记录：实验名、seed、父 checkpoint、GAUC、AUC、LogLoss、峰值显存、QPS、日志路径和决策结论。只填写实际测量值。

- [ ] **Step 5：提交**

```bash
git add config/sim_hard_dev.json config/sim_soft_dev.json docs/results.md
git commit -m "exp: add fixed SIM development baselines"
```

### Task 2：支持公平的训练时 warm-start

**文件：**
- Create: `tests/test_checkpoint_loading.py`
- Create: `config/muse_warmup_dev.json`
- Create: `config/muse_continue_dev.json`
- Modify: `main.py`

- [ ] **Step 1：先写失败测试**

测试 `validate_warm_start_paths(args)`：只提供 dense checkpoint 时抛出异常；dense/sparse 路径任一不存在时抛出 `FileNotFoundError`；两者都存在时返回路径对。

- [ ] **Step 2：确认测试先失败**

```bash
pytest tests/test_checkpoint_loading.py -v
```

Expected：因为函数尚不存在而失败。

- [ ] **Step 3：实现加载顺序**

在 `main.py` 中实现：模型先移动到当前 GPU；校验并加载 dense/sparse warm-start；然后基于已加载参数建立 AdamW/SparseAdam；最后再 DDP 包装。续训不加载 optimizer state，保证各分支以相同学习率公平开始。

- [ ] **Step 4：创建 warm-up 与 continuation 配置**

warm-up：`exp_name=muse_warmup_dev_1pct`、`method=muse`、统一开发协议、`save_ckpt=true`。

continuation：统一开发协议、`exp_name=muse_continue_dev_1pct`、`save_ckpt=false`，并写入：

```json
{
  "warm_start_dense_ckpt": "./ckpt/muse_warmup_dev_1pct_dense.ckpt",
  "warm_start_sparse_ckpt": "./ckpt/muse_warmup_dev_1pct_sparse.ckpt"
}
```

- [ ] **Step 5：测试并运行**

```bash
pytest tests/test_checkpoint_loading.py tests/test_horizon.py tests/test_horizon_gate.py -v
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_warmup_dev.json 2>&1 | tee logs/muse_warmup_dev_1pct.log
test -s ckpt/muse_warmup_dev_1pct_dense.ckpt
test -s ckpt/muse_warmup_dev_1pct_sparse.ckpt
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json 2>&1 | tee logs/muse_continue_dev_1pct.log
```

Expected：checkpoint 非空，continuation 日志明确显示两次加载，训练和评估完成。

- [ ] **Step 6：提交**

```bash
git add main.py tests/test_checkpoint_loading.py config/muse_warmup_dev.json config/muse_continue_dev.json
git commit -m "feat: add fair warm-start continuation protocol"
```

### Task 3：统一 SA-TA 原始相关性函数

**文件：**
- Modify: `model/base_model/layers.py`
- Create: `tests/test_relevance_score.py`

- [ ] **Step 1：写 GSU/ESU 一致性和 padding 测试**

测试必须验证：`raw_relevance_logits` 输出 `[B, heads, L, 1]`；padding 位置为负无穷；`calc_attn_score` 等于 raw logits 在序列维 softmax 后的 head 均值；包含 SCL bias 时两条路径仍逐元素一致。

- [ ] **Step 2：确认旧代码失败**

```bash
pytest tests/test_relevance_score.py -v
```

Expected：旧版没有 `raw_relevance_logits`，且 `calc_attn_score` 忽略多模态 bias。

- [ ] **Step 3：抽取唯一打分 API**

在 `MultiHeadAttV2` 中实现：

```python
def raw_relevance_logits(self, query, fact, mask=None, mm_cosine=None):
    # 每个 head 计算 d_i，并在同一处融合 m_i
    # r_i = b1 * d_i + b2 * m_i + b3 * d_i * m_i
    # padding 使用 -inf
    # 返回 [B, heads, L, 1]
```

`forward` 和 `calc_attn_score` 都只能调用该方法，不能分别保留另一套公式。当前配置只有一个 head，因此 GSU 与 ESU 的相关性分数严格相同；未来多 head 时 GSU 使用 head 均值。

- [ ] **Step 4：验证并提交**

```bash
pytest tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
git add model/base_model/layers.py tests/test_relevance_score.py
git commit -m "refactor: share SA-TA relevance logits"
```

### Task 4：实现 TWIN-inspired CP-MUSE

**文件：**
- Modify: `main.py`
- Modify: `trainer.py`
- Modify: `scripts/model_smoke.py`
- Create: `config/cp_muse_dev.json`

- [ ] **Step 1：写 padding-safe Top-K 失败测试**

`masked_relevance_topk(score, valid_mask, keep_top)` 必须只返回有效位置；有效行为不足 Top-K 时返回 invalid mask 并将后续特征清零。

- [ ] **Step 2：实现 CP 检索**

在 `Trainer.apply_general_search` 的 `@torch.no_grad()` 范围内：

1. 获取目标与 1,000 条历史的 item/category embedding。
2. 拼成 SA-TA 的 query/fact。
3. 计算与 MUSE 相同的目标-历史 SCL cosine。
4. 调用 `dense_model.uni_att_v2.raw_relevance_logits(...)`。
5. 在完整 1K 历史中按 `r_i` 选择 Top-50。
6. 用同一组 indices gather item、category 和 SCL，并将 invalid slot 清零。

Top-K 不反向传播，但同一打分参数通过 ESU 的 CTR loss 学习。

- [ ] **Step 3：创建覆盖配置并测试**

```json
{
  "exp_name": "cp_muse_dev_1pct",
  "method": "cp-muse",
  "collect_retrieval_diagnostics": true
}
```

```bash
pytest tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
```

- [ ] **Step 4：等步数运行**

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/cp_muse_dev.json 2>&1 | tee logs/cp_muse_dev_1pct.log
```

Expected：加载同一 warm-up checkpoint，完成 376/111 步，并报告 CP 与 continuation MUSE 的 GAUC delta。

- [ ] **Step 5：应用 ETA 门槛并提交**

delta `< -0.0005` 时记录跳过 ETA；否则保留 Task 8。

```bash
git add main.py trainer.py scripts/model_smoke.py config/cp_muse_dev.json tests/test_relevance_score.py docs/results.md
git commit -m "feat: add consistent-relevance CP-MUSE retrieval"
```

### Task 5：实现 GroupPool-TA 全历史残差分支

**文件：**
- Create: `model/base_model/longer_lite.py`
- Create: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Modify: `trainer.py`
- Create: `config/group_pool_dev.json`

- [ ] **Step 1：写失败测试**

验证：6 条有效/无效混合历史以 group size 4 处理后形状固定；padding 不进入均值；全 padding 输出为 0；`residual_scale=0` 时分支启用前后的 MUSE 输出逐元素一致。

- [ ] **Step 2：实现 GroupPoolTA**

行为 token 为 `item ID 16D + category ID 16D + SCL 128D = 160D`，通过 Linear 投影到 32D并加入绝对位置 embedding。每 4 条相邻行为做 mask-aware mean，1K 变成 250 token；目标 token 对 250 token 做 target attention，输出 32D 兴趣向量。

分支参数：

```python
self.token_projection = nn.Linear(160, 32)
self.position_embedding = nn.Embedding(1000, 32)
self.target_projection = nn.Linear(160, 32)
self.residual_scale = nn.Parameter(torch.zeros(()))
```

- [ ] **Step 3：接入 MUSE**

只在启用 `longer_variant` 时由 trainer 保留 full-history embedding，并向 `MUSE_DIN.forward` 传入 `full_seq_embs`。分支输出按下式接入，不改变预测塔输入宽度：

```python
uni_seq_att_out_v2 = uni_seq_att_out_v2 + residual_scale * full_history_interest
```

checkpoint 仅在分支存在时保存 `longer_lite`；加载普通 MUSE checkpoint 时保留新分支初始化参数并打印说明，不能报缺 key。

- [ ] **Step 4：测试并运行**

配置覆盖：

```json
{
  "exp_name": "group_pool_dev_1pct",
  "method": "muse",
  "longer_variant": "group-pool",
  "longer_group_size": 4,
  "longer_model_dim": 32
}
```

```bash
pytest tests/test_longer_lite.py tests/test_relevance_score.py tests/test_horizon.py tests/test_horizon_gate.py -v
python scripts/model_smoke.py
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/group_pool_dev.json 2>&1 | tee logs/group_pool_dev_1pct.log
```

Expected：1K -> 250，训练无 OOM，记录参数量、显存、QPS 和排序指标。

- [ ] **Step 5：提交**

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py trainer.py config/group_pool_dev.json docs/results.md
git commit -m "feat: add GroupPool full-history residual"
```

### Task 6：实现 InnerTrans-TA

**文件：**
- Modify: `model/base_model/longer_lite.py`
- Modify: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Create: `config/inner_trans_dev.json`

- [ ] **Step 1：写局部建模失败测试**

验证每组 4 token 只在组内交互，输出 group 数不变，padding 被 key padding mask 屏蔽，全 padding 组输出 0。

- [ ] **Step 2：实现一层共享 InnerTrans**

将 `[B, 250, 4, 32]` 展平为 `[B*250, 4, 32]`，使用一层 `TransformerEncoderLayer(d_model=32, nhead=1, dim_feedforward=64, dropout=0, batch_first=True, norm_first=True)`，然后做 mask-aware pooling。不同 group 之间不能做 self-attention。

- [ ] **Step 3：测试并运行**

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
pytest tests/test_longer_lite.py tests/test_relevance_score.py -v
python scripts/model_smoke.py
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/inner_trans_dev.json 2>&1 | tee logs/inner_trans_dev_1pct.log
```

- [ ] **Step 4：应用 GlobalToken 门槛并提交**

InnerTrans delta `< -0.001` 时跳过 Task 7；否则执行 Task 7。

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py config/inner_trans_dev.json docs/results.md
git commit -m "feat: add InnerTrans local sequence modeling"
```

### Task 7：有条件实现 GlobalToken-LongerLite

仅在 Task 6 门槛通过时执行。

**文件：**
- Modify: `model/base_model/longer_lite.py`
- Modify: `tests/test_longer_lite.py`
- Modify: `model/muse.py`
- Create: `config/global_token_dev.json`

- [ ] **Step 1：写方向性测试**

验证 Cross Attention 的 query 长度为 103（target、user、CLS、最近 100 个 merged token），key/value 长度为全部 250；输出 `[B,32]` 且有限。

- [ ] **Step 2：实现单向读取**

target/user/CLS/recent-100 作为 query 读取全部 250 个历史 token，再对 103 个 query 输出做一层 self-attention并返回 CLS。历史 token 不读取目标或用户 token，保持历史侧可缓存的数据方向。

- [ ] **Step 3：测试并运行**

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

```bash
pytest tests/test_longer_lite.py -v
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/global_token_dev.json 2>&1 | tee logs/global_token_dev_1pct.log
```

- [ ] **Step 4：提交**

```bash
git add model/base_model/longer_lite.py tests/test_longer_lite.py model/muse.py config/global_token_dev.json docs/results.md
git commit -m "feat: explore global-token long-history encoding"
```

### Task 8：有条件实现 ETA-style 哈希粗筛

仅在 Task 4 的 CP-MUSE delta `>= -0.0005` 时执行。

**文件：**
- Create: `utils/hash_retrieval.py`
- Create: `tests/test_hash_retrieval.py`
- Modify: `trainer.py`
- Modify: `main.py`
- Create: `config/eta_cp_muse_dev.json`

- [ ] **Step 1：写失败测试**

验证固定 `seed=42` 的随机投影每次生成相同 32-bit bool code；Hamming Top-200 排除 padding；Top-50 Recall 的分母排除 exact Top-K 中的 invalid slot。

- [ ] **Step 2：实现两阶段检索**

使用任务训练得到的 item/category 表示生成 32-bit code，先按 Hamming 距离取 Top-200，再对 200 个候选使用完整共享 `r_i` 取 Top-50。记录 exact CP Top-50 Recall、哈希耗时、精排耗时、总检索耗时和峰值显存。

- [ ] **Step 3：测试并运行**

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

只有端到端实测耗时低于 exact CP-MUSE 时才能写“加速”；否则只写近似检索的召回-效率权衡。

- [ ] **Step 4：提交**

```bash
git add utils/hash_retrieval.py tests/test_hash_retrieval.py trainer.py main.py config/eta_cp_muse_dev.json docs/results.md
git commit -m "feat: add ETA-style hash retrieval ablation"
```

### Task 9：最终验证、模型选择和项目包装

**文件：**
- Create: `config/final_smoke.json`
- Modify: `scripts/model_smoke.py`
- Modify: `docs/results.md`
- Modify: `README.md`

- [ ] **Step 1：运行完整轻量验证**

```bash
pytest tests -v
python scripts/model_smoke.py
git diff --check
```

Expected：全部通过，无空白错误，无数据、日志或 checkpoint 被覆盖。

- [ ] **Step 2：运行两卡 smoke**

`config/final_smoke.json`：

```json
{
  "exp_name": "final_two_gpu_smoke",
  "max_train_steps": 2,
  "max_eval_steps": 2,
  "save_ckpt": false
}
```

从已经完成的 CP、GroupPool、InnerTrans、GlobalToken 中选实测 GAUC 最高的一份覆盖配置，与 continuation 基础配置和 smoke 覆盖配置按顺序加载。示例：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py --config config/muse_continue_dev.json config/cp_muse_dev.json config/final_smoke.json
```

Expected：两个 rank 都完成 2 个训练和 2 个评估 step并正常退出。

- [ ] **Step 3：决定是否做第二 seed**

最佳模型相对 continuation MUSE 的 delta `< +0.0005`：停止扩展，诚实记录负结果或权衡。

delta `>= +0.0005`：以 `seed=2026` 重新训练一份 MUSE warm-up，再分别运行对应 continuation control 和获胜模型，训练/评估步数严格相同。

- [ ] **Step 4：决定是否跑全量数据**

仅当第二 seed delta `>= 0` 时进入全量。全量配置改用 `./taobao-mm/{train,test,feature_map}`，移除 376/111 步数上限，保留 batch 1000 和 p90 词表。启动前确认额外可用磁盘至少 100GB、内存至少 128GB、两张 24GB GPU 可见；显存不足时优先将 `scl_emb_on_cuda=false`。

- [ ] **Step 5：更新 README 与简历表述**

README 写入：统一协议、完整命令、checkpoint 血缘、结果表、理论依据和限制。若确有提升，简历写真实 GAUC delta、数据规模和对照；若没有提升，写“实现并系统比较一致性检索与压缩式全历史建模”，报告真实的效果-效率权衡，不伪造正收益。

- [ ] **Step 6：最终审计并提交**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

逐个暂存本次新增文件，不使用 `git add config` 吸收旧的未跟踪配置。

```bash
git add README.md docs/results.md scripts/model_smoke.py config/final_smoke.json
git commit -m "docs: report long-sequence recommendation experiments"
```

## 五、最终验收

- [ ] 官方 SIM-hard、SIM-soft、MUSE 和所有改进使用同一开发集与步数协议。
- [ ] CP-MUSE 的 GSU/ESU 调用同一个含多模态 bias 和交叉项的 raw relevance API。
- [ ] padding 不进入 Top-K、GroupPool、InnerTrans、GlobalToken 或 ETA recall 分母。
- [ ] GroupPool 与 InnerTrans 均覆盖完整 1K 历史并输出 250 个 32D merged token。
- [ ] 所有新增分支从同一 warm-up checkpoint 等步数续训。
- [ ] GlobalToken、ETA、第二 seed 和全量实验均遵守停止门槛。
- [ ] 结果记录 GAUC、AUC、LogLoss、显存、QPS/延迟、参数量、seed、日志和父 checkpoint。
- [ ] 所有测试和两卡 smoke 通过后才启动更昂贵的实验。
- [ ] README 明确区分“借鉴/探索”和“完整复现”。
- [ ] 未删除、移动、清理或覆盖任何数据集、checkpoint、日志和用户已有文件。
