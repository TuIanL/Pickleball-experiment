"""
检查跨模型参考输出是否正常：对比 RTMPose-x 与原 RTMPose-M 在同一帧上的核心 8 点（肩肘腕髋）。
用法：python server_check.py --root <包根目录> [--out skeleton/pose_rtmposex]
"""

import argparse
import json
import os

import numpy as np

CORE = [5, 6, 7, 8, 9, 10, 11, 12]
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="skeleton/pose_rtmposex")
a = ap.parse_args()
xd = os.path.join(a.root, a.out)
md = os.path.join(a.root, "skeleton", "pose_coco17")
ids = sorted(f[:-5] for f in os.listdir(xd) if f.startswith("clip_"))
dists, sx, sm, bad = [], [], [], []
for sid in ids:
    X = json.load(open(os.path.join(xd, sid + ".json")))
    M = json.load(open(os.path.join(md, sid + ".json")))
    if len(X["frames"]) != len(M["frames"]):
        bad.append(sid)
        continue
    for fx, fm in zip(X["frames"], M["frames"], strict=True):
        if fx["keypoints"] and fm["keypoints"]:
            px = np.array([[k["x_px"], k["y_px"]] for k in fx["keypoints"]])[CORE]
            pm = np.array([[k["x_px"], k["y_px"]] for k in fm["keypoints"]])[CORE]
            dists.append(np.linalg.norm(px - pm, axis=1))
            sx.append([fx["keypoints"][j]["score"] for j in CORE])
            sm.append([fm["keypoints"][j]["score"] for j in CORE])
d = np.concatenate(dists)
print("片段数:", len(ids), " 帧数不一致:", bad)
print(
    "核心 8 点 RTMPose-x 与 RTMPose-M 像素距离：中位数 %.1f, P90 %.1f"
    % (np.median(d), np.percentile(d, 90))
)
print("平均 score：RTMPose-x %.3f, RTMPose-M %.3f" % (np.mean(sx), np.mean(sm)))
print(
    "判断：中位数应在约 2–10 像素；如果 >30 像素或 score 明显偏低，多半是颜色顺序(--color)或框格式不对，先停下来检查。"
)
