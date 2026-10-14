"""Build one in-memory record per clip from the hand-off package (pose, PTS, A/B and adjudicated labels)."""

import json
import pickle

import numpy as np
import pandas as pd

from .paths import DATA_ROOT, work

JOINTS6 = [
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]
BOUNDARIES = ["backswing_start", "forward_start", "contact_frame", "followthrough_end"]


def _ab_visibility(rec, n):
    out = {j: np.array(["visible"] * n, dtype=object) for j in JOINTS6}
    for it in rec.get("visibility_intervals") or []:
        if it["joint"] in out:
            out[it["joint"]][it["start_frame"] : it["end_frame"] + 1] = it["visibility"]
    return out


def build_clips(root=DATA_ROOT):
    """Return {sample_id: dict}; raises on frame-count or PTS mismatch."""
    idx = pd.read_csv(root / "clip_index_947.csv")
    meta = {}
    for line in open(root / "metadata/clips_947.jsonl", encoding="utf-8"):
        r = json.loads(line)
        meta[r["sample_id"]] = r
    rev, ab = {}, {}
    for line in open(root / "annotations/reviews_879.jsonl", encoding="utf-8"):
        r = json.loads(line)
        rev[r["sample_id"]] = r
    for line in open(root / "annotations/original_AB_947.jsonl", encoding="utf-8"):
        r = json.loads(line)
        ab[r["sample_id"]] = r
    clips = {}
    for _, row in idx.iterrows():
        sid = row.sample_id
        pose = json.load(open(root / row.skeleton_path, encoding="utf-8"))
        n = len(pose["frames"])
        xy = np.full((n, 17, 2), np.nan)
        sc = np.full((n, 17), np.nan)
        bbox = np.full((n, 4), np.nan)
        pts = np.array([f["pts_seconds"] for f in pose["frames"]])
        for f in pose["frames"]:
            i = f["frame_index"]
            for k in f.get("keypoints") or []:
                xy[i, k["index"]] = [k["x_px"], k["y_px"]]
                sc[i, k["index"]] = k["score"]
            if f.get("bbox_source"):
                bbox[i] = f["bbox_source"]
        if n != row.num_frames or not np.allclose(pts, meta[sid]["pts_seconds"], atol=1e-4):
            raise ValueError(f"frame/PTS mismatch: {sid}")
        a_rec, b_rec = ab[sid]["original_records"]
        c = dict(
            sid=sid,
            origin=row.origin,
            stroke=row.source_stroke_type,
            group=row.package_group,
            usage=row.source_usage,
            N=n,
            t=pts,
            xy=xy,
            sc=sc,
            bbox=bbox,
            side=pose["target_subject"]["human_review"].get("target_side"),
            A=a_rec,
            B=b_rec,
            visA=_ab_visibility(a_rec, n),
            visB=_ab_visibility(b_rec, n),
        )
        if sid in rev:
            d = rev[sid]["decision"]
            c["final"] = dict(
                hitting_arm=d["hitting_arm"],
                backswing_status=d["backswing_status"],
                clip_completeness=d["clip_completeness"],
                target_ok=d["target_subject_confirmed"],
                stroke_ok=d["stroke_label_confirmed"],
                bounds=d["boundaries"],
                source="review",
            )
            c["vis"] = {j: np.array(d["visibility"][j], dtype=object) for j in JOINTS6}
        else:
            # Not in the review export: rule-based consensus from the two original annotations.
            bd = {}
            for k in BOUNDARIES:
                va, vb = a_rec["boundaries"][k], b_rec["boundaries"][k]
                bd[k] = (
                    None
                    if (va is None or vb is None)
                    else {"low": min(va, vb), "high": max(va, vb)}
                )

            def agree(key, a_rec=a_rec, b_rec=b_rec):
                return a_rec[key] if a_rec[key] == b_rec[key] else "disagree"

            c["final"] = dict(
                hitting_arm=agree("hitting_arm"),
                backswing_status=agree("backswing_status"),
                clip_completeness=agree("clip_completeness"),
                target_ok=a_rec["target_subject_confirmed"] and b_rec["target_subject_confirmed"],
                stroke_ok=a_rec["stroke_label_confirmed"] and b_rec["stroke_label_confirmed"],
                bounds=bd,
                source="AB_rule",
            )
            vis = {}
            for j in JOINTS6:
                a, b = c["visA"][j], c["visB"][j]
                both_vis = (a == "visible") & (b == "visible")
                both_not = (a != "visible") & (b != "visible")
                vis[j] = np.where(both_vis, "visible", np.where(both_not, a, "disputed")).astype(
                    object
                )
            c["vis"] = vis
        clips[sid] = c
    return clips


def load_clips(path=None):
    path = path or work("clips.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)
