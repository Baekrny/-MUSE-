# GlobalToken 分阶段训练执行状态

更新时间：2026-07-31

## 当前状态

本轮 GPU 实验已完成。代码、配置、单元测试、CPU smoke、双卡 staged smoke、1% 两 seed、10% 和全量正式实验均已完成并同步到 `/root/autodl-tmp/ha-muse`。

开发分支：`codex/global-token-staged`

## 已完成

- 为 LongerLite 增加可配置的 `longer_residual_init`，staged 与 joint 均使用 `0.05`。
- 新增 `StagedLongerController`：stage 1 冻结 MUSE 主干与稀疏模型，只训练 `longer_lite`；到指定 step 后恢复原始 `requires_grad` 状态并切换联合学习率。
- stage 1 长度固定为 75 step，stage 2 为 225 step，总预算 300 step。
- 新增三组正式实验配置：MUSE control、GlobalToken joint、GlobalToken staged。
- 新增两步 staged smoke 配置，用于 GPU 启动后的 DDP 集成检查。
- 核心测试结果：`35 passed in 50.50s`。
- CPU smoke 结果：GroupPool、InnerTrans、GlobalToken 前反向有限值检查通过，stage 1 到 stage 2 转换通过。
- 无卡服务器必须设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；否则 CPU Transformer smoke 会因线程过度调度超过 5 分钟。固定线程后 smoke 在 40.8 秒内完成。
- 双卡 staged smoke 通过，日志确认 branch-only 和 joint 两个阶段均执行。
- 三组正式实验均完成 300 train steps 和 111 eval steps。

## 正式结果

### 全量主结果

| 实验 | GAUC | AUC | LogLoss | 相对 MUSE control |
|---|---:|---:|---:|---:|
| MUSE warm-up | 0.614801 | 0.645007 | 0.383749 | 不适用 |
| MUSE low-LR control | 0.593760 | 0.625307 | 0.411067 | 0 |
| GlobalToken staged | 0.597236 | 0.629558 | 0.403234 | +0.003476 |

全量 staged 相对 matched control 提升 AUC `+0.004251`、降低 LogLoss `0.007833`，但仍低于续训前的 warm-up，不能表述为超过原始 MUSE。

### 1% 机制实验

| 实验 | GAUC | AUC | LogLoss | 相对 MUSE control |
|---|---:|---:|---:|---:|
| MUSE low-LR control | 0.578522 | 0.602126 | 0.394461 | 0 |
| GlobalToken joint | 0.578873 | 0.602621 | 0.394256 | +0.000351 |
| GlobalToken staged | 0.581907 | 0.606298 | 0.391152 | +0.003385 |

GlobalToken staged 在 seed 42 上比 joint 高 `+0.003034`，相对 control 提升 `+0.003385`；在 seed 2026 上相对匹配 control 提升 `+0.003554`。两个 seed 均超过预注册 `+0.0005` 晋级门槛。

## 实验协议

三组实验都从同一个 `muse_warmup_dev_1pct` checkpoint 出发，使用 seed 42、相同数据顺序、300 个训练 step 和完整 111 个评估 step。

| 实验 | 前 75 step | 后 225 step | residual init |
|---|---|---|---:|
| MUSE low-LR control | 全参数，dense `5e-5` / sparse `5e-4` | 不变 | 不适用 |
| GlobalToken joint | 全参数，dense `5e-5` / sparse `5e-4` | 不变 | 0.05 |
| GlobalToken staged | 仅 `longer_lite`，dense `2e-4` | 全参数，dense `5e-5` / sparse `5e-4` | 0.05 |

joint 与 staged 使用相同残差初始化，因此两者主要比较 branch-only warm-up 是否缓解新分支适配不足。三组总训练步数一致。

## GPU 执行记录

先运行两步 DDP smoke，并使用全新日志名：

```bash
cd /root/autodl-tmp/ha-muse
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/ha-muse

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py \
  --config config/muse_continue_short_dev.json \
           config/global_token_staged_300_dev.json \
           config/global_token_staged_smoke.json \
  2>&1 | tee logs/global_token_staged_smoke_<timestamp>.log
```

smoke 必须同时出现以下两条日志：

```text
Staged longer training: branch-only for 1 steps
Staged longer training: joint phase at step 1
```

smoke 通过后依次运行正式实验：

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py \
  --config config/muse_continue_short_dev.json config/muse_low_lr_300_dev.json \
  2>&1 | tee logs/muse_low_lr_300_dev_1pct_<timestamp>.log

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py \
  --config config/muse_continue_short_dev.json config/global_token_joint_300_dev.json \
  2>&1 | tee logs/global_token_joint_300_dev_1pct_<timestamp>.log

OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 main.py \
  --config config/muse_continue_short_dev.json config/global_token_staged_300_dev.json \
  2>&1 | tee logs/global_token_staged_300_dev_1pct_<timestamp>.log
```

运行前先确认对应日志不存在，避免 `tee` 覆盖历史文件。`<timestamp>` 替换为实际启动时间。

## 预注册决策规则

- 主要对照：GlobalToken staged 相对 MUSE low-LR control 的 GAUC。
- 机制对照：GlobalToken staged 是否优于 GlobalToken joint。
- staged 相对 MUSE control 达到 `GAUC +0.0005`，才进入第二 seed 和更大数据验证。
- 未达到 `+0.0005` 时停止扩展，不运行 5%、10% 或全量数据。
- 在 GPU 结果产生前，README 中只能将该方案描述为“待运行实验”，不能声明精度提升。

## 收口状态

1. 10% 匹配验证已完成，staged 相对 control 提升 `GAUC +0.005136`。
2. 全量匹配验证已完成，staged 相对 control 提升 `GAUC +0.003476`。
3. README 和结果文档以全量实验为主结果，1% 与 10% 作为跨 seed、跨规模一致性证据。
4. 简历表述必须明确对照为 matched continuation control，避免写成线上收益或超过原始 MUSE。
