# Pickleball-experiment

[![CI](https://github.com/TuIanL/Pickleball-experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/TuIanL/Pickleball-experiment/actions/workflows/ci.yml)

匹克球单目比赛视频中的骨架轨迹恢复实验仓库。

对应论文题目（已冻结）：**Phase-Aware Skeletal Trajectory Restoration for Reliable Kinematic Analysis from Monocular Match Video in Pickleball**。

当前状态（2026-10-13）：E0–E4 的实验代码已迁入 `src/pickleball_experiment/` 与 `scripts/experiments/`，参数由开发 pilot 冻结（`configs/frozen_params.json`）。运行方法、汇总结果和未完成项见 [docs/experiments/README.md](docs/experiments/README.md)。视频、骨架和人工标注仍只保存在本地。

## 快速开始

建议使用 Python 3.11。以下命令适用于 macOS / Linux：

```bash
git clone https://github.com/TuIanL/Pickleball-experiment.git
cd Pickleball-experiment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

ruff check .
ruff format --check .
python scripts/check_repository.py
```

`requirements-dev.txt` 只固定代码检查工具；运行实验需另装 `requirements.txt`（numpy、scipy、pandas、scikit-learn、matplotlib、pytest）。服务器上的跨模型姿态估计依赖见 `docs/experiments/cross_model_reference_runbook.md`。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/pickleball_experiment/` | 后续实验方法与可复用代码 |
| `scripts/` | 命令行脚本、仓库文件检查 |
| `configs/` | 可版本化的实验配置，不含凭据和正式数据 |
| `tests/` | 后续算法与数据契约测试 |
| `docs/` | 研究边界与复现说明 |
| `data/` | 本地素材、标注、数据清单；内容默认不进 Git |
| `models/` | 本地模型权重；内容默认不进 Git |
| `outputs/` | 本地实验输出与日志；内容默认不进 Git |

## 研究口径

主终点是投影 **2D elbow-angle MAE**，核心比较 **Uniform (U) vs Phase-aware (P)**，动作 / clip 配对差 **Δ = MAE_P − MAE_U**。Phase 使用本研究定义的 movement phases 和同视角人工相位提示；motion prior 指 **phase-specific temporal smoothness prior**。

Reliable 的范围、人工二维参考 H 子集、探索性下游分析和待确定参数见 [研究边界](docs/research_scope.md)。AI 协作约束见 [AGENTS.md](AGENTS.md)。

## 日常开发

```bash
git switch -c feat/your-change
ruff check . --fix
ruff format .
python scripts/check_repository.py
git add <需要提交的文件>
git diff --cached
git commit -m "Describe the change"
git push -u origin HEAD
```

通过 Pull Request 合入 `main`。GitHub Actions 对 push 和 PR 运行 Ruff 与已跟踪文件检查；当前还没有算法测试，不将 CI 通过等同于研究有效性验证。

## 数据与授权

本仓库为公开仓库。视频、逐帧人物素材、人工标注、模型权重、私密清单与实验输出默认保留在本地。需要发布的派生材料应明确其来源、授权和脱敏范围，再单独组织提交；未知伦理、同意与素材授权不得写成已通过。

`.gitignore` 对已经被 Git 跟踪的文件不生效，提交前仍需检查暂存区。数据目录规则见 [data/README.md](data/README.md)。

## 许可证

当前尚未选择开源许可证；公开可见不代表已经授予再分发或其他使用许可。
