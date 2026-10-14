"""
跨模型参考：用 RTMPose-x（384x288）在与现有 RTMPose-M 骨架相同的视频、相同目标框上重新估计 17 个关键点。
输出格式与 skeleton/pose_coco17 完全一致（pb.pose-coco17.v1.0），写入 skeleton/pose_rtmposex/。

用法见《07_服务器跨模型参考_操作方案.md》。
  试跑：python server_cross_model_pose.py --root <包根目录> --limit 3
  全量：python server_cross_model_pose.py --root <包根目录>
"""

import argparse
import json
import os
import time

import av
import numpy as np

DEFAULT_ONNX = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-x_simcc-body7_pt-body7_420e-384x288-71d7b7e9_20230629.zip"
)
NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="WRSSC2026_complete_947_20261004 目录")
    ap.add_argument(
        "--onnx", default=DEFAULT_ONNX, help="RTMPose-x body7 384x288 的 onnx 链接或本地路径"
    )
    ap.add_argument("--out", default="skeleton/pose_rtmposex")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--color",
        default="rgb",
        choices=["rgb", "bgr"],
        help="必须与原 RTMPose-M 运行一致；原索引记录为 rgb",
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个片段（试跑用）")
    a = ap.parse_args()

    from rtmlib import RTMPose

    model = RTMPose(
        onnx_model=a.onnx, model_input_size=(288, 384), backend="onnxruntime", device=a.device
    )

    src_dir = os.path.join(a.root, "skeleton", "pose_coco17")
    out_dir = os.path.join(a.root, a.out)
    os.makedirs(out_dir, exist_ok=True)
    ids = sorted(
        f[:-5] for f in os.listdir(src_dir) if f.startswith("clip_") and f.endswith(".json")
    )
    if a.limit:
        ids = ids[: a.limit]
    t0, fails = time.time(), []
    for n, sid in enumerate(ids):
        outp = os.path.join(out_dir, sid + ".json")
        if os.path.exists(outp):
            continue  # 支持断点续跑
        src = json.load(open(os.path.join(src_dir, sid + ".json")))
        frames_out = []
        with av.open(os.path.join(a.root, "videos", sid + ".mp4")) as cont:
            for fi, fr in enumerate(cont.decode(video=0)):
                img = fr.to_ndarray(format="rgb24" if a.color == "rgb" else "bgr24")
                sf = src["frames"][fi]
                assert sf["frame_index"] == fi, (sid, fi)
                rec = dict(
                    frame_index=fi,
                    pts_seconds=sf["pts_seconds"],
                    bbox_source=sf.get("bbox_source"),
                    bbox_status=sf.get("bbox_status"),
                    pose_status="bbox_missing",
                    keypoints=None,
                )
                if sf.get("bbox_source"):
                    try:
                        k, s = model(img, bboxes=[sf["bbox_source"]])
                        k, s = np.asarray(k)[0], np.asarray(s)[0]
                        rec["keypoints"] = [
                            dict(
                                index=j,
                                name=NAMES[j],
                                x_px=float(k[j, 0]),
                                y_px=float(k[j, 1]),
                                score=float(s[j]),
                            )
                            for j in range(17)
                        ]
                        rec["pose_status"] = "ok"
                    except Exception as e:
                        rec["pose_status"] = "pose_failed"
                        fails.append([sid, fi, repr(e)])
                frames_out.append(rec)
        assert len(frames_out) == len(src["frames"]), (sid, len(frames_out), len(src["frames"]))
        out = dict(
            schema_version="pb.pose-coco17.v1.0",
            sample_id=sid,
            frame_convention=src["frame_convention"],
            video=src["video"],
            coordinate_space=src["coordinate_space"],
            model=dict(
                name="rtmpose-x",
                checkpoint=a.onnx,
                input_size_wh=[288, 384],
                color_order=a.color,
                note="cross-model reference; bboxes copied from pose_coco17",
            ),
            keypoint_layout=src["keypoint_layout"],
            target_subject=src["target_subject"],
            frames=frames_out,
        )
        json.dump(out, open(outp, "w"))
        if n % 50 == 0:
            print(f"{n + 1}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    json.dump(fails, open(os.path.join(out_dir, "_failures.json"), "w"))
    print("done", len(ids), "clips; failures:", len(fails))


if __name__ == "__main__":
    main()
