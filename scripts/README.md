# 脚本

`check_repository.py` 使用 Python 标准库检查 Git 已跟踪文件，阻止默认本地数据目录、视频、模型和部分凭据文件进入公开提交；它不进行文件内容级秘密扫描。

```bash
python scripts/check_repository.py
```

实验脚本在 `experiments/`（按 s00–s14 顺序运行，见 `docs/experiments/README.md`）；服务器上运行的跨模型姿态估计脚本在 `server/`。
