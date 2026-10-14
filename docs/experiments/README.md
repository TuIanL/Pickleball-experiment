# 实验代码与复现

本目录记录 2026-10-13 完成的 E0–E4 实验。数据不在 Git 中：视频、骨架、标注都留在本地数据包 `WRSSC2026_complete_947_20261004`。

| 文件 | 内容 |
| --- | --- |
| `protocol_v2.md` | 最终实验流程（v2），包括对审查意见的逐条处理 |
| `results_2026-10-13.md` | 数据核查、冻结参数和 E0–E4 的汇总结果 |
| `cross_model_reference_runbook.md` | 服务器跨模型参考（RTMPose-x）的操作方案 |

## 环境

```bash
python -m pip install -r requirements.txt
export PB_DATA_ROOT=/path/to/WRSSC2026_complete_947_20261004   # 默认 data/WRSSC2026_complete_947_20261004
export PB_WORK_DIR=outputs                                       # 中间文件与结果，不进入 Git
```

## 运行顺序

所有命令都从仓库根目录运行。

| 脚本 | 内容 | 主要输出（PB_WORK_DIR） |
| --- | --- | --- |
| `s00_build_dataset.py` | 读取骨架、PTS、A/B 原始标注和仲裁结果；核对帧数与 PTS | `clips.pkl` |
| `s01_data_checks.py` | S0：纳排字段和描述统计 | `s0_table.csv` |
| `s02_agreement.py` | S1：两名标注者的一致性 | 终端输出 |
| `s03_occlusion_distribution.py` | E0：各阶段遮挡比例；置信度识别遮挡的 ROC-AUC | `e0_occ.csv` |
| `s04_make_masks.py` | 生成合成遮挡窗口 | `masks_all.csv` |
| `s05_pilot_calibration.py` | 只用开发集确定 λ 范围和阶段倍率 | 终端输出 |
| `s06_reserve_reference_subset.py` | 核对预留的 24 个片段 | 终端输出 |
| `s07_e1_run.py` | E1：所有方法和参数组合 | `e1_S.pkl` |
| `s08_e1_analyze.py` | E1：按来源留一选择参数，主效应和 bootstrap 置信区间 | `e1_summary.pkl` |
| `s09_e1_strata.py` | E1：分层描述 | 终端输出 |
| `s10_e2_boundary_sensitivity.py` | E2：阶段边界敏感性 | `e2.pkl` |
| `s11_reference_confidence.py` | 参考置信度的影响 | 终端输出 |
| `s12_e3_real_occlusion.py` | E3：真实遮挡区间的合理性指标 | `e3.csv` |
| `s13_e4_classification.py` | E4：四类击球识别（探索性） | `e4.csv` |
| `s14_figures.py` | 论文图 1–7 | `figures/` |

`s07` 在 2 核机器上约 5 分钟，`s10` 约 3 分钟。冻结参数见 `configs/frozen_params.json`，预留片段见 `configs/reference_subset_ids.json`。

## 复现核对

用重构后的代码重新运行，结果与 2026-10-13 的原始运行一致：掩码文件逐行相同；E1 三来源等权 MAE 依次为线性插值 13.322°、PCHIP 11.450°、U 10.456°、P 10.158°；Δ(P−U) 为 −0.297°，95% CI 为 [−0.406, −0.202]。

## 2026-10-14 修正（PR #1 审查意见）

- E4 分类训练和评价排除 24 个预留片段（n 由 893 变为 869）。
- 二阶差分矩阵的缓存键改为完整时间轴，避免不同 PTS 误用同一矩阵；重跑后 E1 结果不变。
- E3 跳变比例只统计有定义的角度差值。
- 图 1、图 6 的数值改为读取 `e1_table.csv`、`e2_summary.csv`、`ref_confidence.csv`，不再写死。
- CI 增加安装分析依赖和运行 pytest；新增两项回归测试。

## 未完成

- 跨模型参考：等服务器结果，见 `cross_model_reference_runbook.md`。
- 人工二维参考 H：计划保留，人工点标尚未执行，记录为协议偏离（见 `protocol_v2.md` 末尾）。预留的 24 个片段已从所有分析中排除，包括 E4 分类训练。跨模型参考只是补充证据，不替代 H。
