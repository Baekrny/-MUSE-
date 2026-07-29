# HA-MUSE 一致性检索与压缩式长序列建模设计

## 1. 目标与范围

项目基于 TAOBAO-MM Open-source-1k，系统比较搜索式与压缩式长序列建模。搜索主线以 MUSE 为基础，借鉴 TWIN 的 GSU/ESU 相关性度量一致性；效率侧借鉴 ETA 的 LSH 近似检索；序列建模侧用 LONGER-Lite 验证 Group Pooling、InnerTrans 和条件式 Global Token 方案。

双时域门控保留为已完成的负向消融，不再作为主模型。RRF 双通道融合从主线移除：它不能满足 TWIN 所要求的相同相关性度量，且固定融合常数缺少当前 CTR 场景的直接依据。

## 2. 理论依据与设计约束

- SIM 建立 GSU 粗检索与 ESU 精建模的两阶段范式。
- MUSE 认为高质量多模态余弦足以承担轻量 GSU，而 ESU 需要更丰富的 ID-多模态交互。
- TWIN 指出 GSU 与 ESU 的目标-行为相关性函数不一致会导致候选遗漏，核心要求是两阶段使用相同度量，而不只是共享 embedding。
- ETA 使用 LSH 近似目标注意力，降低长序列端到端相关性计算成本，并减少独立检索目标与 CTR 目标之间的信息鸿沟。
- LONGER 通过相邻 Token Merge、轻量 InnerTrans、Global Token 和压缩 Query 降低长序列 Transformer 的二次复杂度。

TAOBAO-MM 仅提供有序 item/category 序列，没有逐行为时间戳。因此 LONGER-Lite 只使用绝对位置与新旧顺序，不能复现时间差特征。公开序列长度为 1K，任何 ETA/LONGER 效率收益都必须实测，不能外推到 10K 或 100K。

## 3. 统一实验协议

- 1% 用户一致性开发集：训练 755,196 行，测试 227,689 行；
- `seed=42`、`batch_size=1000`、`epochs=1`、`Top-K=50`；
- 两张 RTX 4090、DDP、`OMP_NUM_THREADS=1`；
- `max_train_steps=376`、`max_eval_steps=111`，保证两个 rank 同步；
- GAUC 为主指标，AUC、LogLoss、QPS、检索耗时和峰值显存为辅助指标。

官方 SIM-hard、SIM-soft 和 MUSE 使用完全相同的数据与训练协议，进入主对比表。

CP-MUSE 的检索分数和 LONGER-Lite 分支均含新增参数，不能从随机状态直接与已收敛的 MUSE 比较。主改进实验先训练一轮 MUSE warm-up checkpoint，再从同一 checkpoint 分叉并继续训练一轮：一支保持 MUSE 作为 continuation baseline，其他分支分别启用 CP-MUSE、GroupPool-TA 或 InnerTrans-TA。这样各分支拥有相同训练数据量和优化步数。

## 4. TWIN-inspired CP-MUSE

### 4.1 一致性相关性函数

复用 MUSE SA-TA 的目标-行为打分：

```text
d_i = f_q(q)^T f_k(k_i)
r_i = b_1 * d_i + b_2 * m_i + b_3 * d_i * m_i
```

其中 `m_i` 为目标与历史商品的 SCL 多模态相似度。GSU 使用 `r_i` 在完整 1K 历史中选择 Top-50，ESU 使用同一套投影层、融合参数和 `r_i` 计算注意力。Top-K 不反向传播，但共享打分参数通过 ESU 的 CTR 损失学习。

实现时从 SA-TA 提取唯一的 raw relevance score API，GSU 与 ESU 都调用该 API，避免两个公式随代码演进再次分叉。CP-MUSE 是 TWIN 一致性思想在 1K 数据上的可行性验证，不包含 TWIN 的工业级预计算与缓存系统。

CP-MUSE 从 MUSE warm-up checkpoint 启动，避免随机相关性函数在训练初期检索到低质量候选并形成自强化误差。对照组 MUSE 从同一 checkpoint 继续训练相同步数。

### 4.2 ETA-style 近似检索

ETA-CP-MUSE 使用任务训练得到的 ID+类目基础表示，通过由 `seed=42` 固定的随机投影生成 32-bit 二值码，按 Hamming 距离粗筛 Top-200，再使用完整 `r_i` 选择 Top-50。

哈希版对照无哈希 CP-MUSE，报告 Top-50 Recall、GAUC、耗时和显存。由于离线实现仍需扫描 1K 历史，哈希版只作为条件式效率消融；没有真实加速时不得宣称 ETA 带来性能收益。

## 5. LONGER-Lite 序列建模探索

LONGER-Lite 先作为 warm-started MUSE 的残差全历史分支评估，不立即与 CP-MUSE 组合。行为 token 由 item、category 和 SCL 表示投影到 `2 * embedding_dim = 32` 维，并加入可学习绝对位置 embedding。分支输出一个 32 维兴趣向量，通过初始化为 0 的可学习残差系数加入 MUSE 长期兴趣，因此启用分支前的输出与 warm-up MUSE 完全一致，且无需改变下游塔输入维度。

### 5.1 GroupPool-TA

每 4 个相邻行为组成一组，mask-aware mean pooling 将 1K 序列压缩为 250 个 token，再用目标商品对 250 个 token 做 Target Attention。该版本用于验证“保留全历史覆盖、降低序列长度”的最小方案。

### 5.2 InnerTrans-TA

在每个 4-token 小组内运行一层轻量 InnerTrans，再以 mask-aware pooling 生成 merged token，后续 Target Attention 与 GroupPool-TA 相同。InnerTrans 的作用是补偿简单 pooling 对组内顺序和交互的损失。只要 GroupPool-TA 能稳定完成训练，就继续该实验，以保证至少验证 LONGER 的 Token Merge 与 InnerTrans 两个核心组件。

### 5.3 GlobalToken-LongerLite

仅当 InnerTrans-TA 相对同 seed continuation MUSE 的 GAUC 差值不低于 `-0.001` 时实现。目标商品、用户表示和 CLS 作为 Global Token，连同最近 100 个 merged token 组成 Query，对全部 250 个 merged token 做一层 Cross Attention，再做一层 Self Attention。目标/全局 Query 可以读取历史，历史表示不反向读取目标，以保持可缓存的数据流方向。

若 LONGER-Lite 单独带来增益，再从同一 warm-up checkpoint 测试 CP-MUSE 与最佳 LONGER 分支组合；在此之前不引入额外融合门控。所有 LONGER 变体报告新增参数量、估算 FLOPs 和实测 QPS，避免把单纯增加容量误判为序列结构收益。

## 6. 实验顺序与停止规则

1. 运行官方 SIM-hard、SIM-soft 开发集基线。
2. 生成一轮 MUSE warm-up checkpoint，并运行相同步数的 continuation MUSE 基线。
3. 从 warm-up checkpoint 运行 CP-MUSE，验证一致性检索能否超过 continuation MUSE。
4. 从同一 checkpoint 运行 GroupPool-TA 和 InnerTrans-TA，保证至少完成两级 LONGER 探索。
5. InnerTrans-TA 与 continuation MUSE 的 GAUC 差值不低于 `-0.001` 时，再实现 GlobalToken-LongerLite。
6. ETA 哈希只在 CP-MUSE 与 continuation MUSE 的 GAUC 差值不低于 `-0.0005` 时运行；否则没有值得近似的目标相关检索器。
7. 开发集最佳方案相对 continuation MUSE 提升至少 `0.0005` 时，用第二个 seed 及其对应 continuation baseline 复核；第二个 seed 的 GAUC 差值不低于 `0` 才进入全量训练。

该顺序保证四天内优先完成严格基线、TWIN 一致性主实验和至少一级 LONGER 探索，不为凑模块同时展开所有分支。

## 7. 正确性与最小测试

- padding 不得进入 Top-K、Group Pooling、InnerTrans 或注意力归一化；
- raw relevance score 在 GSU 与 ESU 上对同一输入必须逐元素一致；
- Group Pooling 和 InnerTrans 始终返回固定形状，并正确处理不足 4 条的有效历史；
- ETA 投影由配置 seed 唯一确定，Top-50 Recall 计算必须排除 padding；
- 每项新增行为只增加一到两个针对性测试，再运行现有测试和双卡 smoke；
- 每个实验使用新的日志文件，不覆盖已有数据、日志或 checkpoint。

## 8. 项目表述边界

可表述为：基于 MUSE 探索 GSU/ESU 一致性多模态检索，并比较搜索式 CP-MUSE 与 LONGER-inspired 压缩式全历史建模；进一步评估 ETA-style 哈希近似的候选保持率与效率权衡。

不可表述为：完整复现 ETA、TWIN 或 LONGER，支持 100K 序列，复现截图中的 GAUC 提升，或在没有全量实验结果时宣称显著提升。

## 9. 参考依据

- SIM: https://arxiv.org/abs/2006.05639
- ETA: https://arxiv.org/abs/2108.04468
- TWIN: https://arxiv.org/abs/2302.02352
- LONGER: https://arxiv.org/abs/2505.04421
- MUSE: https://arxiv.org/abs/2512.07216
