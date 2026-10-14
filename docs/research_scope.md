# 研究边界与当前状态

本文记录作者已冻结的口径，不替代完整实施协议，也不新增实验结果。

## 核心问题

题目：**Phase-Aware Skeletal Trajectory Restoration for Reliable Kinematic Analysis from Monocular Match Video in Pickleball**。

保持现有三阶段框架。主要终点为投影二维肘角的 MAE；核心比较 Uniform (U) 与 Phase-aware (P)，动作 / clip 配对差为 **Δ = MAE_P − MAE_U**。负 Δ 表示 P 的 MAE 更低，不预设正式实验一定产生负 Δ。

Reliable 限于遮挡 / 损坏条件下的二维肘角恢复准确性、输出覆盖和时间稳定性；不代表三维真实性、临床有效性或已测定重测信度。

## 相位与先验

Movement phases 为本研究操作性定义；当前使用同视角人工相位提示。不得表述为全自动相位检测或通用生物力学分期。

motion prior 指 **phase-specific temporal smoothness prior**。正则化 / MAP 的数学条件以作者当前协议为准；尚未学习真实轨迹分布，不声称 learned phase-specific trajectory prior。

## 参考与分析范围

- 保留 Human-Annotated Reference Subset (H) 计划。人工二维共识独立于模型，不能称三维 / 解剖真值。
- 60 条开发 pilot 不等同正式 H，也不等同独立外部测试。
- 分类仅为探索性下游验证，角速度峰保真仅为次要分析。
- 不扩大主 RQ 至肩腕、功率或全身指标；负结果如实报告。

## 当前进展（2026-10-13）

- 阈值、λ 网格、阶段倍率和合成遮挡窗口已由开发 pilot 冻结，见 `configs/frozen_params.json`。
- 正式评价使用 605 个片段，按来源留一交叉验证；结果汇总见 `docs/experiments/results_2026-10-13.md`。
- H 计划保留，但人工点标尚未执行（截至 2026-10-14），记录为协议偏离，见 `docs/experiments/protocol_v2.md` 末尾的偏离记录。预留的 24 个片段已从全部分析（含分类训练）中排除，H 完成前不得解除预留。
- 跨模型参考（RTMPose-x）只作为补充证据，不替代 H：它仍是模型估计，不能称人工参考或真值。在 H 完成前，凡依赖 H 的结论都标为未验证。

素材许可、人物帧授权、同意与伦理状态需按已有事实记录；未知状态不能自动改成通过。
