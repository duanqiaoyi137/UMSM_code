# UMSM 学生指南：论文、代码与合作实验

更新：2026-10-29。统一入口见 [README](../README.md)，可复制实验顺序见 [实验手册](RESEARCH_EXPERIMENTS.md)。本指南直接对照本轮 `tex` 中已经写出的内容，不改写论文。

## 1. 论文核对结论与边界

| 论文位置/要求 | 对应实现 | 本轮处理 |
|---|---|---|
| problem.tex：多表列、语义组、每表最多一列、允许单例 | types.py、evaluation.py、global_optimizer/ilp.py | 匹配对仍只展开非单例；`Matching.partition(columns)` 和报告 `partition` 显式给出完整 Ω，包括未匹配单例 |
| uncertainty.tex Step 1：多生成器候选并集 | uncertainty_model/mixture.py `generate_candidates/build_qx` | 保留；不同生成器未返回的候选贡献为 0 |
| Eq generator_distribution：各生成器 softmax(r/T) | mixture.py `build_qx` | 保留；T>0，距离转负分数，正频次转 log vote 后缩放 |
| Eq mixture_distribution：非负权重和为 1 | 同上 | 保留；生成器整体无候选时对剩余可用生成器重新归一化，显式记录 active_weights |
| 自监督随机伪缺失，校准温度/权重 | mixture.py `pseudo_mask_examples/fit_mixture`、training.py | 只隐藏指定拟合集中的观测值，隐藏值不会出现在候选输入；覆盖失败单独统计，不伪造候选 |
| 最小累计质量前缀 ρ、K_max 上限、再归一化 | uncertainty_model/base.py `truncate_and_normalize` | 保留；先混合后截断；不足目标质量也不超过 K_max，记录未达标 |
| 每轮每列只实例化一次，所有关联边复用 | types.py `iter_realizations`、online.py `estimate_online` | **修正**：不再预存 R 份列值；导引表分类抽样保留原种子下的样本，期望 O(1) 每格抽样 |
| f(c_i,c_j) 的随机性只来自两个列的缺失格 | matcher/column_pair.py、research.evidence_scope | **补齐**严格二列适配器；原生整表上下文作为单独协议保留 |
| 每次 s 经固定 g_phi，再求 μ/σ | calibration.py、online.py `OnlineMoments.update` | 保留并在线执行；不是先均分数再做 sigmoid |
| 已校准概率允许 g=identity | calibration.py `calibrate_scores`、training.py | **补齐** `calibration.method=identity`；非法概率拒绝，不悄悄裁剪任意原始分数 |
| Eq estimated_uncertainty：R−1 | online.py `OnlineMoments.finish` | **修正执行方式**：Welford 的 M2/(n−1)，批量实现仅作数学参照/旧基线 |
| clip(μ−λσ,ε,1−ε) | uncertainty_propagation/calibrated.py | 保留；诊断和正式图使用相同 ε |
| Algorithm 1：在线 μ/S，不存全部分数 | online.py、pipeline.py | **修正**：逐轮校准更新；普通推断只返回轻量 `EvidenceSamples`，不再保留 R×E 原始表 |
| Efficient Estimation：复用完整列表示 | embedding.py、sbert_mean.py、sbert_quantile.py | 向量/列表示按内容复用；完整列无需重复编码/聚合 |
| E* 为候选连通分量内全部跨表列对 | candidates.py、global_optimizer/logit.py | 保留；不是只给 blocking 的正边评分；保存的 blocking 元信息也只保留实际候选，避免无关二次方日志 |
| matching.tex / analysis.tex 的全局匹配与两表提纲 | global_optimizer/logit.py、ilp.py、local_merge.py | 保留最新 signed-logit 分区 ILP、同表排斥、三角约束、分量分解、精确/显式近似标记；两表目标与二部图解核对 |

**不能声称已经实现论文中尚未写出的细节。** 当前 `matching.tex` 仍只有图、目标/约束/求解器和 Algorithm 2 的提纲，`analysis.tex` 仍是定理提纲；全局优化的具体公式沿用本项目已经实现的最新版理论。`experiment.tex`、`appendix.tex`、`revision.tex` 还含 MINOR、时序清洗、RMSE 或旧审稿回复模板残留，它们不属于本项目的多表列匹配算法。本轮没有据此添加时序清洗算法，也没有修改 tex。

所有已写出的缺失分布、传播公式和 Algorithm 1 均有执行路径与离线检查；不能把未完成的 Algorithm 2 或模板实验包装成已复现的论文内容。

## 2. 首先区分三个概率

假设某表 `manufacturer` 一个单元格缺失：

- `q_x(Apple)=0.6`：这个单元格取值是 Apple 的候选概率。
- `p_ij^(r)=0.8`：第 r 次实现下，两列对应的校准置信度。
- `p_tilde=clip(mu−lambda*sigma)`：风险调整后的保守证据，不自动是更好的后验概率。

q_x 的熵、列对分数的 σ、匹配 P/R/F1 是不同层次的量。重复补入已有值也可能改变频率，但不一定改善语义判断。

## 3. 现在的代码结构

```text
run.py / python -m umsm      唯一正式实验接口
umsm/cli.py                 参数解析、阶段选择和配置检查
configs/paper.yaml          给实验人员修改的四模块入口
configs/research/           数据角色、三折与原生基线的小型配方
umsm/presets.py             五种缺失策略/四种决策器，只在这里组装一次
umsm/research.py             fit/验证/全局/缺失/表数等研究协议
umsm/isolation.py            无决策边质量、无 BFS 表对决策、PR 前沿、2×2
umsm/pipeline.py             构建四模块，调用模块 1–3
umsm/online.py               Algorithm 1 的逐轮执行、Welford、缓存
umsm/types.py                四模块共享的数据对象与按轮采样
umsm/uncertainty_model/      模块 1：Q 与补全对照
umsm/matcher/                模块 2：列表示、COMA、Magneto 等
umsm/uncertainty_propagation/ 模块 3：校准统计、风险与势
umsm/global_optimizer/       模块 4：ILP 与 local/greedy/ALITE 对照
umsm/training.py             显式拟合；推断不会自己拟合参数
umsm/calibration.py          冻结校准校验、g_phi 和概率诊断
umsm/candidates.py           独立于 Q 的轻量 blocking 与闭包
umsm/report.py              一个 HTML 保存指标、配置、校准与具体错误
umsm/isolated_metrics.py     分层指标、配对簇 bootstrap、交互效应
umsm/research_analysis.py    结构冲突、目标函数、σ 和科学绘图
umsm/release.py              不含私密资产的源码交接包
prepare_*.py                公开来源转换/遮蔽，不是另一套匹配算法
tests/                      数学与协议正确性，不是论文实验数据
```

旧 research.py、ablation.py、fit_calibration.py 是几行转发兼容入口。旧 Companies/SOTAB study、教学 example 和历史数据集试跑 YAML 已移出公开源码；备份只在忽略的 `cache/paper_cleanup_20261029`。数据、模型、冻结校准、评分/LLM 缓存、已有结果没有删除。

## 4. 数据怎样在模块间流动

```text
IntegrationSet(原表、列标识、可选真值)
  ├─ candidates.plan_candidates(原表) → 固定 E* / 分量
  └─ model.estimate(原表) → UncertainData(原观测 + 每格 Q)
       ↓ iter_realizations：当前轮所有列各一个实现
     matcher.score(当前轮快照) → 当前轮 ScoreSamples
       ↓ 固定 g_phi → Welford(count, mean, M2)
     calibrated_logit.from_moments → EdgeTable(mu,sigma,p_tilde,a)
       ↓ optimizer.optimize
     Matching → partition(含单例 Ω) + 指标 + 一个 HTML
```

真值只用于指定拟合集的监督校准或评估，不参与测试候选生成、Q 构造、blocking 或匹配分数。数据角色、文件内容与真值签名在研究协议中检查，改文件不能静默继续使用旧验证选择。

关键类型：

| 类型 | 核心字段 | 作用 |
|---|---|---|
| ColumnRef | table、name、id=`table.column` | 跨文件一致的列标识 |
| IntegrationSet | tables、groundtruth、meta | 一个联合匹配任务 |
| UncertainTable | frame、distributions、filled | 原观测不变；不确定格附 Q，确定补全才写 frame |
| UncertainData | source、tables、realization、seed | 按列/轮共享实现，管理候选允许范围 |
| ScoreSamples | col_i/j、state_i/j、sample_id、prob、score | 当前轮原始分数，或显式批量数学参照 |
| EvidenceSamples | count、n_pairs、meta、可选 raw | 在线推断凭据；默认没有 R 张原始分数表 |
| EdgeTable | mu、sigma、variance、p_tilde、a、w | 固定证据，可交给多个决策器 |
| Matching | groups、info、artifacts | groups 保留非单例表示；partition(columns) 返回完整 Ω |

`score` 可以不是概率；只要有限，Platt 可将它映射到概率尺度。选择 identity 时必须是 [0,1] 内的已校准概率，否则拒绝运行。

## 5. 模块 1：构造 q_x

[mixture.py](../umsm/uncertainty_model/mixture.py) 的主路径：

1. `generate_candidates` 调用配置中的生成器。默认 KNN 同行上下文 + 本列边际频率。
2. 每个生成器先将方向统一为越大越可能的 raw score，再分别 softmax(r_g/T_g)。
3. `build_qx` 在支持集并集上按 alpha 混合。某候选未被某生成器返回时，该项为零。
4. [base.py](../umsm/uncertainty_model/base.py) 的 `truncate_and_normalize` 按概率降序取最短累计质量前缀，至多 K_max 个，再归一化。

论文例子：两生成器 `(Apple,Dell,Sony)=(.6,.3,.1)` 与 `(.2,.7,.1)`，权重 `.7/.3`，混合得 `.48/.42/.10`。ρ=.9 时保留前两项，归一化得 `.5333/.4667`。

拟合只在显式 `fit`：随机隐藏观测单元格，候选生成看不到其原值，优化温度与混合权重的 NLL。内部还有独立伪掩码诊断。覆盖不到隐藏真值时记失败并保留有限 NLL floor，不把真值强行塞进候选。

全空列/无候选格没有凭空可用的信息。实现保留缺失并报告 `n_cells_without_evidence`，不从测试标注制造分布。某生成器完全不可用时，可用生成器权重重新归一化；这两项是论文未展开的显式边界处理。

## 6. 模块 2：严格二列与原生整表要分开

论文写的是 `P_ij` 由两列缺失格集合 M_ij 的乘积分布诱导。如果其他列的采样能改变 `(i,j)` 分数，σ 就不再只来自 M_ij。

- SBERT mean / fixed quantile 等列表示方法本来就是列对局部函数。
- COMA、Magneto 原生模式可能使用整表归一化、检索候选或 reranking 上下文。即使两个目标列完整，其他列采样也可能让分数变化。
- [column_pair.py](../umsm/matcher/column_pair.py) 把输入限制为正在比较的两列，使函数严格符合 f(c_i,c_j)。这是适配后的独立方法，不称为原生整表 Magneto/COMA 的同一实验。

`research.evidence_scope=pair` 自动适配整表 matcher；`table` 保留原生整表协议。两种模式不能混用校准，`check_matcher` 会检查实际构造参数和评分版本。主配置选 pair；已有 `isolate_*.yaml` 沿用 table，方便复现历史结果及原生 observed-only 优化器对照。

全表上下文 Q 的 σ 可作为额外扩展研究，但应解释为上下文也在变化的证据，而非仅由两列缺失诱导。原生 observed-only 无采样变化，适合固定同一套 Magneto 边证据比较 local/ILP。

SBERT mean 保留值频次、对行序稳定；quantile 方法须用 `normalization=fixed` 才能保证分数不因其他实现的极值变化。`legacy_minmax` 只保留为明确标记的批量原始方法参照，不能在 Algorithm 1 在线模式中使用。

## 7. Algorithm 1：在线执行的细节

阅读 [online.py](../umsm/online.py) 的 `estimate_online`：

1. 固定 E* 和冻结 g，检查缓存；每条边预先知道应有的实现数量。
2. `UncertainData.iter_realizations` 每轮只生成当前列值，所有关联边共享同一个列实现。各缺失格独立分类抽样，不构造列状态笛卡尔积。
3. matcher 在当前轮快照上评分，代码检查不能漏边、加边或重复某条边。
4. 每个原始 s 先经过 g，得到 p，再更新：

```text
n ← n + 1
delta ← p − mean
mean ← mean + delta/n
M2 ← M2 + delta*(p − mean)
sigma ← sqrt(max(M2,0)/(n−1))
```

5. 对完全确定、可复用的列对，只评分一次，σ=0；这与重复 R 次相同值的结果等价。原生整表模式则在相关表有变化时保留所有上下文轮次。
6. 所有边的实际次数必须等于预期次数；无穷分数、非均匀 MC 权重、缺边、重复轮内边都拒绝。
7. `from_moments` 统一计算风险和势，避免在线和批量参照重复写一套公式。

分类抽样预处理每格 O(K)，导引表逆 CDF 抽样期望 O(1)，并保持原向量化采样的随机数顺序和样本。列表示缓存不算在论文额外空间界内。普通推断不存 R×E 分数，Profiler 也合并重复轮次记录。

额外空间界适用于在线推断工作区，排除原始输入、模型/列表示缓存和显式调试输出。拟合阶段的 Platt 训练、旧 pickle 首次兼容导入、`save_score_samples=true` 的全量诊断不是该空间结论的测量路径。

还有一处理论表述边界：softmax/混合/截断至少需要扫描原始候选支持集。如果论文的 K_bar 仅指截断后保留数量，完整构造 q_x 的成本不能只用它表示，还应计原始候选数量及排序/截断成本；代码不能在不读原始候选的情况下精确计算这些概率。这里的空间结论针对截断后的采样/统计工作区，不宣称消除了候选生成和归一化的成本。这一复杂度文字尚需在论文中明确，本轮不改 tex。

## 8. 模块 3：风险调整与边势

```text
p^(r) = sigmoid(gamma*s^(r) + beta)       或显式 identity
mu = mean_r p^(r)
sigma = sample_std_r p^(r), denominator R−1
p_tilde = clip(mu − lambda*sigma, epsilon, 1−epsilon)
a = logit(p_tilde) − logit(prior)
```

λ 控制不确定性惩罚；prior 决定零势基准，改变边势和匹配数量。二者不是同一个参数。λ=0 仍保留数值裁剪；它不意味着跳过 Platt。不能把原始 cosine 当成已校准概率，除非明确声明并验证校准前提。

`mean_std/expected` 是原始分数对照，不是论文的校准主方法。用它们时应选择兼容的 raw 优化器，例如 `ilp`，而不是需要 signed-logit 输入的 `ilp_logit`。

## 9. 模块 4：最新全局分区

`LogitILP.optimize` 检查每个候选分量内跨表闭包完整，负边也必须有分数。`PartitionILP` 为分量内列对建二元变量 y：

```text
maximize sum a_ij*y_ij
y_ij = 0 if i,j belong to the same table
y_ij + y_jk − y_ik ≤ 1
y_ij + y_ik − y_jk ≤ 1
y_ik + y_jk − y_ij ≤ 1
```

小分量可一次加入全部三角约束；大分量 lazy 模式反复加入违反的三角，直到解传递一致。默认要求精确最优；若显式允许近似，会在报告中标记，不能称为精确全局最优。

`recover_groups` 用选中的边恢复分组，并检查每组同表最多一列且内部为完整团。未选边的列通过 `partition` 作为单例保留。这里的连通恢复是 ILP 可行解的解释，不是局部 Hungarian 后的 BFS 启发式。

对照：local=逐表对 dummy Hungarian 后 BFS；greedy=贪心合并；ALITE-style=同一证据上的聚类。`pairwise` 评估直接接受的表对边，不做任何多表分区，因此不会输出伪造的 Ω。

## 10. 拟合、验证、测试和缓存

| 角色 | 能读取的真值 | 能改变的参数 |
|---|---|---|
| fit | 明确声明的校准任务；Q 用隐藏观测值，Platt 用对应标签 | T、alpha、gamma、beta；identity 跳过 Platt 拟合 |
| validation | 独立验证任务 | prior 或独立的 λ 验证选择 |
| test | 仅计算指标和错误分析 | 不再拟合、不按测试成绩选参数 |
| diagnostic_test_fit | 测试标签，只用于显式诊断 | 仅当前诊断报告的 Platt，不发布正式校准 |

推断的校准快照和选择报告带 SHA256；候选范围、MC 配置、matcher 或校准改变，不能静默借用旧验证选择。严格 pair 协议从源头改变 f，必须独立拟合。现有 table 协议的温度、均值/方差公式不因改成在线执行而改变，只有浮点累计顺序可能有舍入差异。

缓存有三层：

1. 新 `cache/evidence`：保存 μ/σ 等统计，换 λ/prior/优化器无需评分。
2. 可选 `cache/score_rounds`：每轮一个原始分数文件，换 Platt 时按轮重放，不重新询问 matcher。实际 Q 被哈希，不因仅校准文件名不同而无谓失效。
3. 原 `cache/scores/*.pkl`：首次只读导入，随后使用小型统计缓存；原文件不改写。LLM responses 和 embedding 缓存继续保留。

校准/数据/候选/评分语义变了而不能复用，是正确的失效。不要通过改 hash 绕过检查。记录中的缓存命中耗时不能冒充端到端冷启动速度。

`qx_from` 只复用 Q 的温度和权重，允许跨 matcher；新的 Platt 始终按当前 matcher 重新拟合。验证选择比较的是实际评分参数，允许仅更换运行并发或缓存开关；改变评分语义不能继续借用该选择。

## 11. 实验结果怎么读

- edges：同一闭包上的 μ/p_tilde，按两端最大自然缺失率分箱；AUC/AP 越高好，Brier/NLL 越低好。
- pairwise：直接接受边的 TP/FP/FN/P/R/F1；闭包外真边计 FN，不做传递闭包。
- missing 固定 local：完整聚合流程下的补全效果，属于补充证据。
- global/frontier：固定 observed-only 边证据，比较优化器或预声明 prior 网格。
- factorial：{observed,Q}×{local,ILP}；交互为 `(Q_ILP−obs_ILP)−(Q_local−obs_local)`。

AP 提升不保证 Hungarian/ILP F1 提升，ILP 目标上升也不保证真值 F1 上升。正交互不表示联合方法必然超过基线。每折仅一个任务时，列 bootstrap 是固定预测下的条件区间；重叠表子集/多 seed 不当作独立任务。

默认每次只写一个 HTML。`predictions` 中可以看到接受边、分组、完整 partition 和具体 FP/FN；`analysis_tables` 给出分层、簇 CI、PR 前沿、交互与目标差。保存更大诊断是显式开关，不会额外产生一堆 CSV。

## 12. 以后增加方法应改哪里

| 要增加什么 | 最小改动 | 接口 |
|---|---|---|
| 新候选生成器 | mixture.GENERATORS 注册一个工厂 | generate(frame,table,iset,ctx) → 候选 raw scores |
| 新 Q 模型 | 继承 DistributionModel 或 DeterministicModel 并注册 | estimate → UncertainData |
| 新列 matcher | 继承 PairwiseMatcher 或 BaseMatcher 并注册 | 当前轮 data → ScoreSamples；概率校准交给模块 3 |
| 原生整表 matcher | 声明 sampling_scope=table_pair | 原生 table 协议；pair 协议由适配器限制可见列 |
| 新风险/传播 | 继承 UncertaintyPropagation 并注册 | ScoreSamples → EdgeTable；论文主实现为 calibrated_logit |
| 新分组方法 | 继承 GlobalOptimizer 并注册 | 固定 EdgeTable → Matching |
| 新公开数据 | 适配为每源 CSV、groundtruth.txt、provenance.json | 在配置中声明不重叠的 fit/validation/test |

构造参数不要放内部缓存；BaseMatcher.params 只返回可复现参数。缓存使用 `_` 前缀，评分语义变化需提高评分版本，使旧校准明确失效。新传播/优化器必须约定匹配的边字段，不能假设所有模块组合都数学兼容。

离线测试保留 exact 小样本、枚举最优值、同表/三角约束、原生上下文、严格二列、独立数据角色与 API mock。它们用于验证论文实现，不是论文成绩；无需上传虚拟环境或下载模型来运行这些 mock 检查。
