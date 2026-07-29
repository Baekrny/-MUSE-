# HA-MUSE 双通道长序列检索设计

## 1. 目标与范围

本阶段将项目主线从效果不稳定的双时域门控，调整为 ETA-style 哈希粗筛、MUSE 多模态语义检索和 TWIN-inspired 表示一致协同检索组成的三阶段长序列建模方案。目标是在 TAOBAO-MM Open-source-1k 设置下，提高长历史 Top-K 候选的互补性，分析效率与效果权衡，并使用官方实现的 SIM-hard、SIM-soft 和 MUSE 建立严格可比的基线矩阵。

双时域门控保留为消融实验，不再作为主结果。本项目借鉴 ETA 的二值哈希粗筛和 TWIN 的 GSU/ESU 表示一致性，但不声称完整复现 ETA 或 TWIN。TAOBAO-MM 的公开序列长度为 1K，因此 ETA-style 模块主要用于验证候选保持率与检索开销的权衡，不能据此声称已经支持十万级行为序列。

## 2. 实验协议

所有开发集实验固定使用：

- 1% 用户一致性子集：训练 755,196 行，测试 227,689 行；
- `seed=42`、`batch_size=1000`、`epochs=1`、`Top-K=50`；
- 两张 RTX 4090、DDP、`OMP_NUM_THREADS=1`；
- `max_train_steps=376`、`max_eval_steps=111`，保证两个 rank 的 collective 严格同步；
- GAUC 为主指标，AUC、LogLoss、QPS 和峰值显存为辅助指标。

第一阶段直接运行官方 `SIM-hard` 和 `SIM-soft`，并与已完成的官方 MUSE 基线比较。只有相同数据、步数和超参数下的结果进入主表。

## 3. 三阶段检索与精排

### 3.1 ETA-style 哈希粗筛

将冻结的 128 维 SCL 多模态向量通过由 `seed=42` 生成的固定高斯随机投影转换为 32-bit 二值码。使用目标商品与历史商品二值码的 Hamming 距离，从最多 1000 条行为中选择语义 Top-200 候选池。固定投影不参与训练，padding 和缺失 SCL 向量不得被当作有效语义候选。该粗筛只限制高维语义通道，不能限制协同通道。

哈希模块的主要评价指标为：相对于无哈希双通道检索的最终 Top-50 候选保持率、单批检索耗时和峰值显存。GAUC 用于确认粗筛没有造成不可接受的下游损失。首轮只使用 32-bit、Top-200，避免在开发集上无边界搜索参数。

### 3.2 语义通道

沿用 MUSE：在语义 Top-200 候选池上使用目标商品与历史商品的冻结 SCL 多模态向量计算精确余弦相似度，得到语义 Top-50 排名。无哈希消融直接在完整 1K 历史上执行该通道。

### 3.3 协同通道

沿用 SIM-soft：在完整 1K 历史上使用目标与历史的 ID、类目 embedding 计算余弦相似度，得到协同 Top-50 排名。检索与 ESU 共享同一套 ID、类目 embedding 和组合方式，借鉴 TWIN 的 GSU/ESU 表示一致性。该模块可表述为 `TWIN-inspired representation-consistent retrieval`，但不得表述为完整复现 TWIN。即使商品缺失 SCL 向量，协同通道仍有机会将其召回。

### 3.4 排名融合

两路排名使用加权 Reciprocal Rank Fusion 合并：

```text
score(i) = w_mm / (c + rank_mm(i)) + w_cf / (c + rank_cf(i))
```

默认 `w_mm=w_cf=1`、`c=60`。RRF 只依赖名次，不依赖两类相似度的数值尺度，能够稳定保留两路重合候选并吸收各自独有候选。融合后去重并选择最终 Top-50；有效候选不足时沿用现有零填充和 invalid mask 机制。若等权版本接近或超过 MUSE，只额外测试 `2:1` 和 `1:2` 两组权重。

最终候选继续进入 MUSE 的 SimTier 与 SA-TA，不新增第三套精排网络。完整数据流为：

```text
1K history -> ETA-style hash -> semantic pool Top-200 -> MUSE semantic Top-50
           -> full-history SIM-soft collaborative Top-50
           -> RRF Top-50
           -> MUSE SimTier + SA-TA
```

## 4. 实验顺序与决策规则

1. 运行官方 SIM-hard、SIM-soft 开发集基线。
2. 实现并运行无哈希的等权 RRF 双通道模型，隔离融合本身的贡献。
3. 增加 32-bit、Top-200 的 ETA-style 哈希粗筛，测量候选保持率、耗时、显存和 GAUC。
4. 仅当等权模型接近或超过 MUSE 时，再测试语义偏重和协同偏重两组权重；避免无边界调参。
5. 只有开发集 GAUC 稳定超过 MUSE，才进入全量数据训练；若哈希版本 GAUC 略低但候选保持率和速度收益明确，则作为效率消融而非最佳效果模型。

主消融包括：MUSE-only、SIM-soft-only、无哈希等权双通道、ETA-style Hash + 等权双通道，以及最佳权重双通道。SIM-hard 作为规则检索下界。

## 5. 正确性与异常处理

- 候选融合必须确定性去重，并始终返回固定形状的索引和 invalid mask。
- 历史 padding 不得进入有效候选；SCL 缺失项不得进入语义哈希池，但允许协同通道补充。
- 固定随机投影必须由配置 seed 唯一确定，并随模型配置记录，保证哈希结果可复现。
- 增加少量单元测试，覆盖哈希确定性、padding、候选保持率计算、重复候选和候选不足。
- 本地测试通过后运行双卡短 smoke，再运行完整开发集。
- 每个实验使用独立日志文件，不覆盖已有日志或数据。

## 6. 项目表述边界

可表述为：基于 MUSE 构建 ETA-style 哈希粗筛、多模态语义检索与 TWIN-inspired 表示一致协同检索，通过 RRF 融合候选并使用 SA-TA 精排，在 TAOBAO-MM 上系统对比 MUSE、SIM-hard 和 SIM-soft。

不可表述为：完整复现 ETA、完整复现 TWIN、支持 100K 行为序列，或在没有全量实验数据时宣称显著提升。
