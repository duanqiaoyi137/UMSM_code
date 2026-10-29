# 统一实验操作手册

更新：2026-10-29。所有命令在 UMSM 根目录运行；统一接口 `python run.py`，每次评估一个 HTML。本文取代历史数据集专用脚本和旧试跑命令。论文核对与代码解释见 [学生指南](STUDENT_GUIDE.md)。

<a id="isolated"></a>

## 1. 先选协议，不要混合评分语义

| 配置 | 用途 | 校准/报告 |
|---|---|---|
| configs/paper.yaml | 对照 tex 的严格列对主协议，默认 mean | 新 `paper_{matcher}_{fold}.json`，研究名 paper |
| configs/research/isolate_alaska.yaml | 复现既有 Alaska 原生整表 COMA 等隔离实验 | 既有 alaska_* 校准及 alaska_isolated 报告目录 |
| configs/research/isolate_sotab.yaml | SOTAB 多独立任务 | sotab_* 校准 |
| configs/research/isolate_madi.yaml | MaDI games/music/papers | madi_* 校准 |
| configs/research/isolate_gdc.yaml | 原生 Magneto observed-only＋优化器 | gdc_* 校准 |

原生整表 Q 是上下文扩展；不能将它的 σ 解释为论文中仅来自两列的缺失。不覆盖旧文件来切换协议。`column_pair` 适配 COMA/Magneto 后的打分函数改变，必须新校准；不能称为原生基线复现。

## 2. 环境、数据和检查

```powershell
.\.umsm\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py --check-config
python tests/run_all.py
```

已安装依赖可以跳过安装。COMA 的 stopwords 检查见 README；本地已有就跳过下载。首次跑预训练 matcher 需要本地模型或下载，不属于这些离线检查。

数据未准备时才执行：

```powershell
python prepare_alaska.py
python prepare_sotab.py
python prepare_gdc.py
foreach ($domain in @('games','music','papers')) {
    python prepare_madi.py --domain $domain --max-rows 300
}
foreach ($k in @(5,8,12)) {
    python prepare_sotab.py --name "SOTAB_K$k" --tables-per-set $k --grouping distinct_sites --roles test
}
```

来源：Alaska 官方 research.alaska；SOTAB V2 CPA；MaDI-Bench 官方 base 任务；GDC-SM Zenodo 14963588 及论文临床源。脚本固定/记录下载版本与哈希。GDC 正常自动下载，不要复制 `E:/your_download/...` 占位路径。只有已取得全部官方 CSV 的手动回退才使用 `--source-dir`。

转换结果是适合“每表最多一列”的派生源对源任务，不直接冒充官方原任务成绩。多目标源列排除，同表别名按已记录规则选一列；审核 provenance.json。SOTAB_K5/K8/K12 只更换 test，fit/validation 继续来自 SOTAB_CPA，不能降低隔离条件凑数据。

## 3. 严格论文协议：按顺序运行

以下以默认 mean 为例。`--matcher coma` 自动使用严格二列 COMA 适配器和独立的 paper_coma 校准；其计算会更贵，先用 mean 检查流程。

```powershell
$cfg = 'configs/paper.yaml'
$matcher = 'mean'
$fold = 'camera8_notebook8_monitor8'
$artifact = "calibration/research/paper_${matcher}_${fold}.json"

python run.py -c $cfg --matcher $matcher --phase profile
if (-not (Test-Path $artifact)) {
    python run.py -c $cfg --matcher $matcher --phase fit
    if ($LASTEXITCODE -ne 0) { throw 'fit failed' }
}
python run.py -c $cfg --matcher $matcher --phase qcheck
python run.py -c $cfg --matcher $matcher --phase validate-prior
if ($LASTEXITCODE -ne 0) { throw 'prior validation failed' }

$choice = Get-ChildItem "results/research/paper/$fold/$matcher/validate-prior/*.html" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$prior = $choice.FullName

python run.py -c $cfg --matcher $matcher --phase edges --selection $prior -s research.bootstrap.unit=column
python run.py -c $cfg --matcher $matcher --phase pairwise --selection $prior -s research.bootstrap.unit=column
python run.py -c $cfg --matcher $matcher --phase missing --selection $prior -s 'research.optimizers=[local]'
python run.py -c $cfg --matcher $matcher --phase global --selection $prior
python run.py -c $cfg --matcher $matcher --phase frontier --selection $prior
python run.py -c $cfg --matcher $matcher --phase factorial --selection $prior
```

按阶段解释：profile 只读概况；fit 冻结 Q/Platt；qcheck 看拟合集之外的伪掩码；validate-prior 只用验证集 observed-only 的无 BFS Hungarian 选一个共享 prior。后六步分别评价边概率、直接表对决策、local 端到端、固定证据的四优化器、PR 前沿、2×2 交互。

Alaska 每折一个任务，使用 column CI；它条件于当前固定预测，不重新拟合或求 Hungarian。多个独立任务用 task CI。每个阶段有同一套 μ/p_tilde 的分层概率表；Q_mean/Q_risk 的 Q 与 μ/σ 会校验一致。

λ 默认预先固定 0.5，所有策略共享 Platt，所有缺失策略共享 observed-only 选出的 prior。不要用旧 ILP 选 prior 的报告代替新 pairwise 选择。`--selection latest` 在隔离协议中只检索当前研究目录的 validate-prior；需要严格留档时使用上面保存的确切文件名。

切换已有相同 mean 校准时，可显式 `-s research.calibration=现有路径`，但必须通过 matcher、版本和数据来源检查。严格 COMA/Magneto 适配器不能借用原生整表校准。

## 4. 三折和批量执行

下面定义三个 job，以原生 COMA 复现已完成的隔离实验；它们保留旧校准、目录和语义。若改为严格论文新实验，将配置改为 `configs/paper.yaml`，校准和 Folder 前缀相应改为 paper，不能只改一半。

```powershell
$jobs = @(
    @{ Args=@('-c','configs/research/isolate_alaska.yaml','--matcher','coma');
       Artifact='calibration/research/alaska_coma_camera8_notebook8_monitor8.json';
       Folder='results/research/alaska_isolated/camera8_notebook8_monitor8/coma'; Unit='column' },
    @{ Args=@('-c','configs/research/isolate_alaska.yaml','--matcher','coma',
        '-s','research.fold=notebook8_monitor9_camera8',
        '-s','research.roles.fit.sets=[notebook_fit]',
        '-s','research.roles.validation.sets=[monitor_validation]',
        '-s','research.roles.test.sets=[camera_test]');
       Artifact='calibration/research/alaska_coma_notebook8_monitor9_camera8.json';
       Folder='results/research/alaska_isolated/notebook8_monitor9_camera8/coma'; Unit='column' },
    @{ Args=@('-c','configs/research/isolate_alaska.yaml','--matcher','coma',
        '-s','research.fold=monitor9_camera8_notebook7',
        '-s','research.roles.fit.sets=[monitor_fit]',
        '-s','research.roles.validation.sets=[camera_validation]',
        '-s','research.roles.test.sets=[notebook_test]');
       Artifact='calibration/research/alaska_coma_monitor9_camera8_notebook7.json';
       Folder='results/research/alaska_isolated/monitor9_camera8_notebook7/coma'; Unit='column' }
)

function Invoke-Stage {
    param([string[]]$ResearchArgs,[string]$Phase,[string[]]$Extra=@())
    python run.py @ResearchArgs --phase $Phase @Extra
    if ($LASTEXITCODE -ne 0) { throw "Research phase failed: $Phase" }
}

function Initialize-Job {
    param([hashtable]$Job)
    if (-not (Test-Path $Job.Artifact)) { Invoke-Stage $Job.Args 'fit' }
    Invoke-Stage $Job.Args 'validate-prior'
    $choice=Get-ChildItem ($Job.Folder+'/validate-prior/*.html') -ErrorAction Stop |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $choice) { throw 'No prior selection report' }
    $Job.Prior=$choice.FullName
}

function Invoke-Core {
    param([hashtable]$Job)
    if (-not $Job.Prior) { throw 'Initialize the job first' }
    $selected=@('--selection',$Job.Prior)
    $cluster=@('-s',('research.bootstrap.unit='+$Job.Unit))
    Invoke-Stage $Job.Args 'edges' ($selected+$cluster)
    Invoke-Stage $Job.Args 'pairwise' ($selected+$cluster)
    Invoke-Stage $Job.Args 'missing' ($selected+@('-s','research.optimizers=[local]'))
    Invoke-Stage $Job.Args 'global' $selected
    Invoke-Stage $Job.Args 'frontier' $selected
    Invoke-Stage $Job.Args 'factorial' $selected
}

foreach ($job in $jobs) { Initialize-Job $job }
foreach ($job in $jobs) { Invoke-Core $job }
```

每个阶段失败即停止，不会拿上一次成功报告悄悄继续。只想跑第一折就用 `Invoke-Core $jobs[0]`。旧真实成绩不需要因为在线累计重写；在线/批量数值只可能有浮点舍入差，缓存导入也不重算真实 matcher。

## 5. 表数扫描、风险联合消融与校准诊断

```powershell
foreach ($job in $jobs) {
    Invoke-Stage $job.Args 'subsets' @('--selection',$job.Prior,
        '-s','research.global_missing=[observed_only]','-s','research.optimizers=[local,global]')
    Invoke-Stage $job.Args 'factorial' @('--selection',$job.Prior,'-s','research.factorial_q=Q_risk')
}
```

默认 k=2/3/4/6/8/12/all 中可用的 k，每个 k 最多 20 个不同子集；小空间枚举、全表一次。相同子集配对，不按真值重抽。k=2 检查目标相等；并列最优可导致选边/F1 不同。重叠子集不是独立任务。

诊断必须单独运行，不能混进正式成绩：

```powershell
foreach ($job in $jobs) {
    Invoke-Stage $job.Args 'diagnostic-calibration' @('--selection',$job.Prior)
}
```

它在测试标签上拟合 Platt，只保存到标记 TEST-FITTED 的 HTML，不改校准 JSON，不允许用作 selection。测试内拟合也不是保证 F1 更高的上界。

`validate-lambda` 保留为独立的 ILP 验证协议，不把其测试内收益当作概率模块无全局优化的主证据。λ=0、prior 网格与校准迁移是不同对照，不在看过测试结果后随意拼接选择。

## 6. 固定 Q 的跨域/同域校准

```powershell
$crossArtifact='calibration/research/alaska_coma_camera8_notebook8_monitor8.json'
$cross=@{ Args=@('-c','configs/research/isolate_alaska.yaml','--matcher','coma',
    '-s','research.fold=cross_monitor','-s',"research.calibration=$crossArtifact",
    '-s','research.roles.validation.sets=[monitor_validation]');
    Artifact=$crossArtifact; Folder='results/research/alaska_isolated/cross_monitor/coma'; Unit='column' }
$same=@{ Args=@('-c','configs/research/isolate_alaska.yaml','--matcher','coma',
    '-s','research.fold=same_monitor','-s','research.roles.fit.sets=[monitor_fit]',
    '-s','research.roles.validation.sets=[monitor_validation]','-s',"research.qx_from=$crossArtifact");
    Artifact='calibration/research/alaska_coma_same_monitor.json';
    Folder='results/research/alaska_isolated/same_monitor/coma'; Unit='column' }
Initialize-Job $cross
Initialize-Job $same
Invoke-Core $cross
Invoke-Core $same
```

同一个 monitor_test、同一个 Q、不同 Platt 拟合来源。两套各自在同一验证任务选 prior，套内所有缺失策略共用；两个结果不是独立任务。严格 pair 迁移同理，但须给 pair 方法自己的源 artifact、fold 和研究名。

## 7. SOTAB / MaDI / GDC

下面复用第 4 节函数；原生协议用于继续已有研究。比较新论文严格 pair 版本时，另用研究名和校准文件，并设置 `research.evidence_scope=pair`，不要覆盖原生结果。

```powershell
foreach ($matcher in @('mean','coma')) {
    foreach ($dataset in @('sotab','madi')) {
        $fold=if ($dataset -eq 'sotab') { 'websites' } else { 'external_calibration' }
        $job=@{ Args=@('-c',"configs/research/isolate_$dataset.yaml",'--matcher',$matcher);
            Artifact="calibration/research/${dataset}_${matcher}_${fold}.json";
            Folder="results/research/${dataset}_isolated/$fold/$matcher"; Unit='task' }
        Initialize-Job $job
        Invoke-Core $job
    }
}
```

SOTAB 多任务、MaDI 三领域用 task CI；只有少数任务的区间很不稳定，必须保留每任务结果。共享原表的任务用 `research.bootstrap.task_groups` 聚为同一来源簇；一个有效簇时 CI 为 null。

SOTAB-k：沿用 SOTAB_CPA 拟合/验证，分别指定 `-s research.roles.test.name=SOTAB_K5`（K8/K12 同理），并更换 `research.name` 报告目录。不同 k 数据可能共享网站，不能合起来当独立任务。

GDC 原生 Magneto，先只验证优化器，不需要跑所有 Q 实现：

```powershell
python check_llm.py -c configs/magneto_deepseek.yaml
$gdc=@{ Args=@('-c','configs/research/isolate_gdc.yaml','--matcher','magneto',
              '-s','paths.raw_score_cache=true');
    Artifact='calibration/research/gdc_magneto_external_calibration.json';
    Folder='results/research/gdc_isolated/external_calibration/magneto'; Unit='column' }
Initialize-Job $gdc
Invoke-Stage $gdc.Args 'global' @('--selection',$gdc.Prior)
Invoke-Stage $gdc.Args 'frontier' @('--selection',$gdc.Prior)
```

这会在尚无评分/校准时调用付费 API；缓存有效时优化器/prior 扫描不重问模型。完整 Q 及严格二列 Magneto 会改变请求数量与提示，不默认保证便宜。确认需要该对照后才执行 `Invoke-Core $gdc`；该命令仍是原生整表上下文 Q 扩展，严格 pair 需独立配置/校准。

## 8. 自然缺失、受控遮蔽与 oracle

主结果用自然缺失。机制分析可在真实稠密表上生成有 pre-mask oracle 的派生任务：

```powershell
python prepare_variants.py --dataset MaDI --sets games_pilot300 --name Games_Mechanisms `
    --modes mcar mar mnar --rates 0.2 0.4 0.6 0.8 --rows 10 30 100 300 --seeds 42 43 44 `
    --dense-max-missing 0.2 --complete-rows --min-rows 10

python prepare_variants.py --dataset MaDI --sets games_pilot300 --name Games_Transplant `
    --modes transplant --rows 30 100 300 --seeds 42 43 44 `
    --dense-max-missing 0.2 --donor-dataset Alaska --donor-sets monitor_test

python run.py -c configs/research/isolate_madi.yaml --matcher mean --phase missing `
    -s research.roles.test.name=Games_Mechanisms -s "research.roles.test.sets=['*']" `
    -s 'research.optimizers=[local,global]'
```

MCAR 独立遮蔽，MAR 依赖同行保留的驱动列，MNAR 依赖取值，transplant 用真实供体掩码。原自然缺失不当作新遮蔽。脚本保存 seed、精确坐标、oracle 与哈希；恢复比只在 oracle−observed F1>0 时定义。

若稠密筛选后行/表不足，脚本会拒绝；不根据真值重抽到“更有利”的实例。oracle 是遮蔽前参考，不是测试选参数。机制分析不能替代自然缺失主结果。

## 9. 多人同步实验和交接

1. 协调者确定唯一配置、数据 provenance、校准 SHA、R/seed 和验证选择；校准只拟合一次，再分发只读文件。
2. 每人独立目录解压源码，不要同时修改同一份运行中的源码。严格复现时共享协调者导出的依赖锁定版本，结果也会记录环境。
3. 同一方法比较缺失策略时固定 matcher/Platt/prior；不同 matcher 独立校准。配置中 `modules` 的最终解析结果会嵌入 HTML。
4. 每位实验人员用自己的输出/缓存路径；不把测试任务或 seed 当作独立训练参数来源。

```powershell
python run.py -c configs/paper.yaml --check-config --matcher mean
python run.py -c configs/paper.yaml --phase edges --selection path/to/the_shared_validation.html `
    -s experiment.output_dir=results/worker_a -s paths.cache_dir=cache/worker_a `
    -s research.bootstrap.unit=column
python run.py --phase package --output dist/umsm_source.zip
```

共享验证报告不要求位于当前结果目录，但其任务签名、候选范围、实际 matcher/MC 和校准哈希必须一致。不要把 `path/to/...` 当成已有文件直接复制；替换为协调者实际分发的验证 HTML 路径。

ZIP 自动包含源码清单，排除环境、数据、模型、校准、缓存、结果与真实 `.env`。需要复用付费缓存时由协调者另行提供；不要删旧缓存来“清理代码”。准备数据和凭据说明在 README。

## 10. 查看与只读汇总

```powershell
python run.py --phase summarize --reports 'results/research/paper/*/*/edges/*.html' --output results/paper_edges.html
python run.py --phase summarize --reports 'results/research/paper/*/*/pairwise/*.html' --output results/paper_pairs.html
python run.py --phase summarize --reports 'results/research/paper/*/*/frontier/*.html' --output results/paper_frontiers.html
```

不评分、不重算 API；保留报告/任务/prior 分隔，不用测试网格最大 F1 作为正式单点。诊断报告不能混入正式汇总。重复运行会分别列出，正式交接时指定确切文件。

HTML 的关键对照：Q_mean−observed 检验补全，Q_mean−top1 检验保留分布，Q_risk−Q_mean 检验风险，固定证据下 global−local 检验优化器。AUC/AP 越高好、Brier/NLL 越低好；F1 是 0–1，差值乘 100 才是百分点。看具体新增/丢失 TP、FP，而不只看宏均值。

一个高缺失箱的 AP 上升不能替代整体匹配收益；风险收缩不保证概率质量改善；ILP 最优仅指当前势函数最优。报告中的 null CI 表示簇数/有效样本不足，不是“没有误差”。
