# Pickleball-experiment 协作约束

## 作者冻结的研究边界

- 题目逐字冻结为 **Phase-Aware Skeletal Trajectory Restoration for Reliable Kinematic Analysis from Monocular Match Video in Pickleball**。未经作者明确解除冻结，不改题、不主动建议改题。
- 主 RQ 与三阶段框架不扩大。主要终点为 2D elbow-angle MAE；核心比较 Uniform (U) vs Phase-aware (P)；动作 / clip 配对差 Δ = MAE_P − MAE_U。
- Reliable 仅指遮挡 / 损坏条件下投影二维肘角的恢复准确性、输出覆盖与时间稳定性，不等同三维生物力学真实性、临床有效性或已测定重测信度。
- Phase 指本研究操作性 movement phases；当前方案使用同视角人工相位提示，不能声称全自动或已通过生物力学验证的通用分期。
- motion prior 精确定义为 phase-specific temporal smoothness prior。可按当前协议解释正则化先验 / MAP，但未学习真实轨迹分布时不能称 learned phase-specific trajectory prior。
- 保留 Human-Annotated Reference Subset (H) 计划。人工共识是独立于模型的二维视觉参考，不是解剖 / 三维真值。60 条开发 pilot 不能充当正式 H 或独立外部测试。
- 分类仅为探索性下游验证；角速度峰保真仅为次要分析。不扩大肩腕、功率或全身主问题，不为 expected 结果编造增益，诚实报告负结果。
- 不无证据声称“首次”或“无人研究”。未知伦理、同意、素材与人物帧授权不得填成通过。
- 核心口径冻结，但正式样本 ID、H 实际数量、阈值 / 超参数仍待 pilot 确定，不将计划写成已运行实验或完整正式协议已冻结。
- 权威方案来自作者本地 `writing_plan_20260919_v03/README.md`，本仓库尚未迁入该方案包。历史 v0.1 / v0.2 不能覆盖当前口径；缺失数学条件时不自行补成正式协议。
- 更新方案另存版本并同步入口，保留历史、原始数据与人工标注，不覆盖已有标注或从正式测试结果反推规则。

## 仓库工作方式

- 当前为基础框架，方法代码与正式结果尚未迁入。
- Python 基线为 3.11；执行 `ruff check .`、`ruff format --check .` 和 `python scripts/check_repository.py`。
- 数据、人物帧、标注、权重、输出和凭据不进入公开提交；默认本地目录为 `data/`、`models/`、`outputs/`。
- 只在真实代码迁入时添加对应依赖和有意义的测试，不把空框架检查称为实验验证。
