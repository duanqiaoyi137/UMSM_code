# UMSM

含缺失值的多表列匹配框架：构造缺失值候选分布，逐轮传播到匹配概率，综合匹配均值和波动，最终求解全局一致的列分组。

本仓库保存算法实现、运行配置、数据准备脚本和离线测试。数据、预训练模型、校准文件、缓存、实验结果和论文文档需另行准备。

## 目录

| 路径 | 内容 |
| --- | --- |
| `umsm/uncertainty_model/` | 候选生成、按类型混合与校准、统计插补及 Multiple MICE |
| `umsm/matcher/` | S-BERT、Magneto、ALITE 风格列表示、Valentine 及列对适配器 |
| `umsm/uncertainty_propagation/` | 匹配分数校准、均值与标准差、风险调整及边权 |
| `umsm/global_optimizer/` | 全局 ILP、Local、Greedy 和其他分组对照 |
| `configs/` | YAML 配置及研究作业，`base:` 在同目录内继承 |
| `tests/` | 不下载模型、不调用真实 LLM API 的离线测试 |
| `prepare_*.py` | 原始数据转换、划分、受控缺失和 provenance |
| `run.py` | 统一命令行入口；`python -m umsm` 等价 |
| `SOURCE_MANIFEST.json` | 本次源码文件的 SHA256 清单 |

旧入口 `research.py`、`ablation.py`、`fit_calibration.py` 保留兼容转发；根目录其他分析脚本用于汇总和可视化运行时结果。

## 安装和离线检查

建议在 Python 3.11 的独立环境中安装依赖。Linux 示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py --list
python run.py -c configs/full_pipeline_sbert_fix.yaml --check-config
python tests/run_all.py
```

Windows PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`。CUDA 运行需要安装与服务器环境兼容的 PyTorch。

## 四个模块与配置

当前修复版 S-BERT 研究入口是 [configs/full_pipeline_sbert_fix.yaml](configs/full_pipeline_sbert_fix.yaml)，继承 [configs/full_pipeline.yaml](configs/full_pipeline.yaml) 的按类型候选生成配置。

1. 根据观测数据推断列类型，使用同表信息生成候选；混合权重及分布参数在 fit 数据的伪缺失值上学习。
2. 按 Q 抽样生成不同补全版本，交给指定 Matcher。主线配置 `R=32`，确定性列对可以只计算一次。
3. 每轮分数经冻结 Platt 校准，再计算均值 μ 和标准差 σ。Q-risk 使用 `clip(μ−λσ)`，边权为 `logit(p_tilde)−logit(prior)`。
4. 在同表互斥和传递一致性约束下，使用ILP求解列分组。`full` 和 `lazy` 是两种三角约束处理方式。

`Q_mean` 令 λ=0；Top1 对每格取 Q 中概率最大的候选。Multiple MICE 是独立的多重插补基线，当前实现先平均原始 Matcher 分数再校准。风险系数 λ 与 prior 的选择规则由各研究配置指定，不通过测试标签调参。

## 数据 模型与正式运行

数据默认放在 `data/`，本地编码模型放在 `models/`。按需要执行 `prepare_alaska.py`、`prepare_alaska_partitions.py`、`prepare_sotab.py`、`prepare_gdc.py`、`prepare_madi.py` 或 `prepare_variants.py`；具体数据源及参数可查看相应脚本的 `--help`。部分研究使用独立的 fit、validation、test 数据，评估前需要相应数据和自己的冻结校准文件。

例如，准备完配置要求的数据及模型后，可运行 S-BERT 研究中的 GDC 作业：

```bash
python run.py -c configs/full_pipeline_sbert_fix.yaml --phase study --study-part gdc
```

这会按配置执行对应阶段，并在 `output/` 生成校准、状态及结果。正式运行前先用 `--check-config`，并核对 YAML 中的角色划分、模型路径、生成器路由、采样预算和输出位置。已有结果及冻结校准不随源码发布。

Magneto 的 `column_pair` 适配只暴露被评分的两列，与原生整表检索／多候选重排是不同运行方式，必须使用各自配置和校准。不要把“可运行的配置”理解为对应实验已经完成。

## API 凭据

需要 LLM 时，复制 `.env.example` 为本地 `.env`，填写自己的环境变量。进程环境变量优先。提供商设置见 [configs/magneto_deepseek.yaml](configs/magneto_deepseek.yaml)，可用 `python check_llm.py -c configs/magneto_deepseek.yaml` 检查；该命令可能实际调用提供商接口。

