"""Phases, reference eligibility and the projected 2D elbow angle."""

import numpy as np

IDX = {"left": (5, 7, 9), "right": (6, 8, 10)}  # COCO-17 shoulder, elbow, wrist
OCCLUDED = ["occluded_self", "occluded_other"]


def mid(bounds, key):
    """Adjudicated boundaries may be intervals; use the floored midpoint."""
    x = bounds.get(key)
    return None if x is None else (x["low"] + x["high"]) // 2


def phases(c, bounds=None, mode="bfce"):
    """Per-frame phase: 0 outside the stroke, 1 backswing, 2 forward swing/contact, 3 follow-through."""
    if bounds is None:
        keys = ["backswing_start", "forward_start", "contact_frame", "followthrough_end"]
        bounds = {k: mid(c["final"]["bounds"], k) for k in keys}
    n = c["N"]
    ph = np.zeros(n, int)
    f, cf = bounds["forward_start"], bounds["contact_frame"]
    if f is None or cf is None or cf < f:
        return None
    b, e = bounds["backswing_start"], bounds["followthrough_end"]
    if mode == "fc":
        b, e = 0, n - 1
    b = 0 if b is None else b
    e = n - 1 if e is None else e
    ph[b:f] = 1
    ph[f : cf + 1] = 2
    ph[cf + 1 : e + 1] = 3
    return ph


def torso_scale(c, thr=0.3):
    xy, sc = c["xy"], c["sc"]
    ok = (sc[:, [5, 6, 11, 12]] >= thr).all(1)
    if ok.sum() < 3:
        return np.nan
    ms = (xy[ok, 5] + xy[ok, 6]) / 2
    mh = (xy[ok, 11] + xy[ok, 12]) / 2
    return float(np.median(np.linalg.norm(ms - mh, axis=1)))


def elbow_angle(s, e, w):
    """Projected 2D interior elbow angle in degrees (180 = straight)."""
    u, v = s - e, w - e
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def arms(c):
    a = c["final"]["hitting_arm"]
    if a == "both":
        return ["left", "right"]
    return [a] if a in ("left", "right") else []


def ref_ok(c, arm, sthr=0.3, lthr=0.2):
    """Frames usable as reference: adjudicated visible, score >= sthr, limb lengths >= lthr * torso."""
    i = IDX[arm]
    s = torso_scale(c)
    vis = np.all([c["vis"][f"{arm}_{j}"] == "visible" for j in ("shoulder", "elbow", "wrist")], 0)
    sc = c["sc"][:, list(i)]
    xy = c["xy"][:, list(i)]
    ok = vis & (sc >= sthr).all(1) & ~np.isnan(xy[:, :, 0]).any(1)
    if not np.isfinite(s):
        return ok & False
    ua = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1)
    fa = np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1)
    return ok & (ua >= lthr * s) & (fa >= lthr * s)
