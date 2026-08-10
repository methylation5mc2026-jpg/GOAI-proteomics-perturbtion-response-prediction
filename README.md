# GOAI 赛道三：机制约束虚拟细胞

本仓库公开一套面向蛋白质组扰动响应预测的可复现工作流，覆盖数据质控、评分环境、表格模型、分子表征、知识图谱、机制先验、折外集成与测试队列离线自评。竞赛原始数据受赛事协议约束，不在仓库中再分发。

## 当前结果

| 队列 | 模型 / 基线 | 总分 | 说明 |
|---|---|---:|---|
| 验证队列 | 最终分层收缩集成 | `0.508655` | 相对按官方分组实现的均值基线 `0.445443` 提升 `0.063212` |
| 测试队列 | 冻结的 Step-6 提交模型 | `0.463638` | 相对批次上下文基线 `0.424883` 提升 `0.038755` |

测试队列数字是将本项目离线评分口径应用于已释放真值所得的自评结果，**不是官方成绩**。组委会的最终排名使用独立内部评测集；测试侧控制匹配与聚合口径也未完全公开。

需要同时注意三项限制：

- 所有神经网络成员仅使用随机种子 `42`，没有进行多种子重复实验。
- 测试队列的 Module 1 中，最终模型为 `0.834850`，仍低于零变化控制锚点的 `0.848392`。
- 当前得分模型尚未使用菌株基因组特征；1011 酵母基因组项目与 SGD S288C 参考基因组仅列为下一阶段计划资源。

数值来源分别为 [`step6_model_scores.json`](evidence/step6_model_scores.json) 与 [`step7_test_selfscore.json`](evidence/step7_test_selfscore.json)。

## 评审快速核查

1. 先看上面的结果边界，再看 [36 页方案报告](docs/report/GOAI_virtual_cell_preliminary_report.pdf) 的方法、负结果与限制。
2. 用 [公开证据索引](evidence/manifest.json) 校验 70 份 JSON/CSV 快照的大小和 SHA-256；索引本身使 `evidence/` 共 71 个文件。
3. 运行 `uv run python workflow/11_test_harness.py`，独立执行 57 项评分口径自检。
4. 运行 `uv run --no-project --python 3.12.10 --with pillow --with pypdf python tools/validate_release.py`，检查公开范围、隐私、报告与证据完整性。

这些入口支持“结论与评分实现核查”，不等同于无需数据的端到端模型重训；完整重训还需要赛事四文件、外部资源访问和相应算力。

## 仓库结构

```text
workflow/            73 个科学工作流脚本、1 个共享路径模块和 3 个串联驱动
evidence/            70 份 JSON/CSV 结果快照与 1 份确定性完整性索引
figures/             17 幅报告图件
docs/report/         36 页匿名方案报告及同步的 LaTeX 源文件
pyproject.toml       Python 3.12.10 与直接依赖声明
uv.lock              完整锁定依赖图
```

`evidence/` 用于不重跑模型时核对报告数字；体积较大的中间矩阵、预测文件、模型权重、缓存与日志不随仓库发布，但生成它们的脚本均保留在 `workflow/`。

## 数据与运行约定

将赛事提供的四个输入文件放入仓库根目录下的 `input/`：

```text
WAYB_WAYC_metadata_train_val.csv
WAYB_WAYC_metadata_test.csv
WAYB_WAYC_proteome_raw_train_val.csv
WAYB_WAYC_proteome_raw_test.csv
```

工作流也兼容浏览器重复下载产生的 `metadata_train_val(1).csv` 与 `metadata_test(1).csv` 后缀；缺少输入时会一次列出全部可接受文件名，不会自动下载或生成替代数据。

核心约定：

- 全局随机种子为 `42`。
- 官方划分由 `split_final` 冻结；参照统计量只在 `split_final == "train"` 上拟合。
- 缺失值全程保留为 NaN，不做填补；损失与评分均使用有限交集掩码。
- 提交矩阵使用 log2 尺度，列宽为 5,244（样本 ID + 5,243 个蛋白）。
- 运行产物写入 `data/`、`results/`、`logs/`，这些目录默认不纳入版本控制。

安装锁定环境并先运行评分自检：

```bash
uv sync --frozen
uv run python workflow/11_test_harness.py
```

自检包含 57 项，与 SciPy、scikit-learn 参考实现及解析解交叉核对。完整流程按脚本编号执行；长链路可使用：

```bash
bash workflow/run_step5_chain.sh
bash workflow/run_step5_chain2.sh
bash workflow/run_step6_chain.sh
```

修订响应阶段的入口为：

```bash
uv run python workflow/50_selfscore_test.py
uv run python workflow/51_external_data_manifest.py
uv run python workflow/52_figures_revision.py
```

`50_selfscore_test.py` 会先验证测试真值未参与模型、权重或超参数选择；被评分的预测矩阵已在该自评脚本产生前冻结。

持续集成还执行 Python 编译、Ruff 正确性规则、Bandit 中高风险扫描、ShellCheck、锁定依赖漏洞审计、秘密扫描和发布/隐私校验。GitHub Actions 与工具版本均固定；`main` 只接受这些检查通过的更改。

## 外部资源与许可

外部资源的来源、已实际留存的版本信息、访问日期、许可边界、用途和覆盖率记录在 [`step7_external_data_manifest.json`](evidence/step7_external_data_manifest.json)。清单明确区分“直接访问的服务”“经其他服务返回的交叉引用”“文献启发的手工编码先验”和“计划但未使用的资源”。未在原始产物中留存的精确发布号会标为未记录，不作追溯猜测。

代码按 [Apache License 2.0](LICENSE) 发布。公开仓库只含推断靶点效力向量的构建/验证代码与统计摘要，不含完整向量、ChEMBL 活性行或靶点映射表；若重新生成并再分发这些衍生产物，必须遵守 ChEMBL 的 CC BY-SA 3.0。官方竞赛数据不再分发。

完整方案见 [`docs/report/GOAI_virtual_cell_preliminary_report.pdf`](docs/report/GOAI_virtual_cell_preliminary_report.pdf)。
