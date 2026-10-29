# UMSM

带缺失值的多表**列匹配**：候选分布 q_x → 逐轮匹配与在线概率统计 → 全局一致分区。

所有实验统一使用 `python run.py`。实验人员主要修改 [configs/paper.yaml](configs/paper.yaml) 中的四个 `modules`，不需要修改算法源码。`python -m umsm` 等价；旧的 research.py/ablation.py/fit_calibration.py 只保留兼容转发，不再维护第二套解析逻辑。

## 安装与检查

在 UMSM 根目录使用 Python 3.10+；已有 `.umsm` 就直接激活：

```powershell
.\.umsm\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --list
python run.py --check-config
python tests/run_all.py
```

Linux/macOS 使用 `source .umsm/bin/activate`。`--list`、`--check-config` 和离线测试不下载模型、不调用 API。实际运行 SBERT 需要本地模型或首次下载；COMA 需要 NLTK English stopwords。先检查是否已有，不要重复下载：

```powershell
python -c "from nltk.corpus import stopwords; print(len(stopwords.words('english')))"
```

数据转换仍由 `prepare_alaska.py / prepare_sotab.py / prepare_madi.py / prepare_gdc.py / prepare_variants.py` 负责，它们不是另一套算法入口。来源、自然缺失、角色和转换哈希保存在 provenance.json。

## 最短正式流程

默认配置是 Alaska 的独立 fit/validation/test 源划分，使用 SBERT mean 和论文严格的列对证据。**校准文件不存在时才执行 fit**；新方法必须使用自己的校准文件，不覆盖旧文件。

```powershell
python run.py --phase profile
python run.py --phase fit
python run.py --phase validate-prior
python run.py --phase edges --selection latest
python run.py --phase pairwise --selection latest
python run.py --phase global --selection latest
python run.py --phase factorial --selection latest
```

`fit` 生成可复用冻结校准；每次评估只生成一个自包含 HTML，含配置、校准哈希、指标及具体 FP/FN。新默认研究名为 `paper`，不会覆盖 `alaska_isolated` 的已有结果。


## 切换方法

```powershell
python run.py --check-config --matcher coma
python run.py --phase fit --matcher coma
python run.py --phase validate-prior --matcher coma
python run.py --phase pairwise --matcher coma --selection latest
```

`--matcher` 的 mean/name/value/coma/jaccard/distribution/magneto 是预设。完全自定义时，修改 YAML 的 `modules.matcher` 并设置唯一的 `research.matcher` 实验标签。其余三个模块也在同一个 YAML 中选择；模块替换后不会继承不兼容的旧构造参数。

注意证据范围：论文公式假定 f(c_i,c_j) 仅依赖这两列。`paper.yaml` 的 `research.evidence_scope: pair` 会给整表 matcher 加上 `column_pair` 适配器；这与原生整表 COMA/Magneto 是**不同方法，需要独立校准**。原生整表实验继续使用 `configs/research/isolate_*.yaml`，或明确选择 `evidence_scope: table`，并独立命名研究/校准。

## 缓存与计时

默认按轮采样、逐次校准、Welford 更新，不保留 R 张分数表。新缓存保存 μ/σ；改变 λ/prior/优化器可以复用。可设置 `paths.raw_score_cache=true`，按轮缓存原始分数到磁盘，便于付费 matcher 续跑和换校准。旧缓存仍只读兼容，不删除、不改写。

旧单文件 pickle 的首次导入会临时加载整张旧分数表，报告标记 `legacy_cache_import`，不能把该过程当作论文在线空间测量。`--no-cache` 禁用磁盘评分/向量缓存；同一运行内仍复用列表示。`experiment.save_score_samples=true` 是显式的大型调试输出，不用于在线空间复杂度结论。

## API 凭据与交接

复制 `.env.example` 为自己的 `.env`，只在本地填写 `DEEPSEEK_API_KEY`；进程环境变量优先。提供商和模型设置在 [configs/magneto_deepseek.yaml](configs/magneto_deepseek.yaml)。不要向源码、YAML、论文结果或共享包写入 key。配置中的费用仅是用户填写的估算，不作为服务商价格。

```powershell
python check_llm.py -c configs/magneto_deepseek.yaml
python run.py --phase package --output dist/umsm_source.zip
```

