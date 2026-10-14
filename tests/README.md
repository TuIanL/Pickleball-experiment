# 测试

`test_core.py` 用合成小样检查肘角计算、二阶平滑修复（直线运动在缺口内应被精确恢复）、插值不外推以及阶段划分。不使用真实人物数据。

```bash
python -m pytest -q tests
```
