# 服务器跨模型参考：操作方案

**目的**：用更大的姿态模型 RTMPose-x（输入 384×288），在与现有 RTMPose-M 骨架**相同的视频、相同的目标框**上重新估计 17 个关键点。结果作为论文中的第二种参考，用来替代人工点标。
**预计用时**：环境配置约 30 分钟，正式运行不到 1 小时（约 2.8 万帧，单卡 GPU）。

---

## 1. 准备
- **服务器**：一张 NVIDIA GPU，显存 ≥ 8 GB 即可，比如 AutoDL 上的 3090 或 4090，Python 3.10。可以直接用之前跑 RTMPose-M 的那台。
- **需要上传到服务器的数据**：只要整理包里的两个文件夹，共约 1.3 GB，并保持原有的目录结构：
  ```
  WRSSC2026_complete_947_20261004/
    videos/                  # 947 个 mp4
    skeleton/pose_coco17/    # 947 个 json（框从这里读取）
  ```
- **需要上传的脚本**：本仓库 `scripts/server/` 下的 `cross_model_pose.py` 和 `check_cross_model.py`。

## 2. 安装环境
```bash
pip install rtmlib==0.0.16 av numpy
pip install onnxruntime-gpu   # 版本要与服务器的 CUDA 匹配；之前跑 RTMPose-M 的环境可以直接复用
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 输出里必须有 CUDAExecutionProvider，否则会退回到 CPU 上运行，速度很慢
```

## 3. 确认模型能下载
```bash
python -c "from rtmlib import RTMPose; RTMPose(onnx_model='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_420e-384x288-71d7b7e9_20230629.zip', model_input_size=(288,384), backend='onnxruntime', device='cuda'); print('ok')"
```
- 打印出 `ok` 就可以继续。
- 如果下载失败：到 MMPose 的 RTMPose 模型库页面，找到 **RTMPose-x, body7, 384×288** 的 ONNX 包，手动下载并解压。之后运行脚本时加上参数 `--onnx /本地路径/end2end.onnx`。
- 记下所用 onnx 文件的 SHA-256（`sha256sum end2end.onnx`），写论文时要用。

## 4. 试跑 3 个片段并检查
```bash
cd /path/to/WRSSC2026_complete_947_20261004
python cross_model_pose.py --root . --limit 3
python check_cross_model.py --root .
```
- **结果正常**：核心 8 点与 RTMPose-M 的像素距离中位数大约在 2–10 像素，两边的平均 score 相近。
- **结果异常**：如果中位数超过 30 像素，或者 score 明显偏低，多半是颜色顺序不对。先删掉 `skeleton/pose_rtmposex/`，改用 `--color bgr` 重新试跑。**颜色顺序必须与原来跑 RTMPose-M 时一致**（原索引记录的是 rgb）。

## 5. 全量运行
```bash
nohup python cross_model_pose.py --root . > rtmposex.log 2>&1 &
tail -f rtmposex.log
```
- 每完成 50 个片段打印一次进度。
- 运行中断后直接重新执行同一条命令即可，已完成的片段会自动跳过。

## 6. 跑完之后检查
```bash
ls skeleton/pose_rtmposex/clip_*.json | wc -l    # 应为 947
cat skeleton/pose_rtmposex/_failures.json        # 应为 []
python check_cross_model.py --root .                  # 全量再检查一遍，把输出截图或复制下来
```

## 7. 交付
1. 把 `skeleton/pose_rtmposex/` 打包成 zip（约 70 MB），解压到
   本地数据包的 `skeleton/pose_rtmposex/`（不进入 Git）。
2. 再附一份简短记录，写明以下几项：
   - GPU 型号；
   - onnxruntime 版本；
   - 模型链接或本地文件名，以及 onnx 的 SHA-256；
   - `--color` 参数取值；
   - 总运行时间；
   - `check_cross_model.py` 的输出。

## 注意事项
- **不要修改**原来的 `skeleton/pose_coco17/` 和 `videos/`。
- **不要重新做目标检测**。脚本会直接读取原骨架文件里的 `bbox_source`，这样两个模型之间唯一的差别就是姿态模型本身。
- 时间宽裕的话，可以再用一个不同架构的模型（例如 ViTPose-H）按同样的思路再跑一遍，作为第三种参考。这一步不是必须的。
