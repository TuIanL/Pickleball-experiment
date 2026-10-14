# 脚本

`check_repository.py` 使用 Python 标准库检查 Git 已跟踪文件，阻止默认本地数据目录、视频、模型和部分凭据文件进入公开提交；它不进行文件内容级秘密扫描。

```bash
python scripts/check_repository.py
```

后续迁入的实验脚本应记录输入契约、输出位置、依赖与可复现运行命令。
