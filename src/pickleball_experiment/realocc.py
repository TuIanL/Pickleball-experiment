"""Restoration of whole clips with human-labelled occluded frames treated as missing (E3/E4)."""

import numpy as np

from .config import SEL_P
from .kinematics import IDX, OCCLUDED
from .restoration import interp_restore, smooth_restore


def restore_clip(c, arm, ph, method, thr=0.3):
    """Interior gaps are filled; edge gaps keep the raw estimate for every method."""
    i = IDX[arm]
    t, n = c["t"], c["N"]
    a = np.ones(n)
    if ph is not None:
        a[ph == 1] = SEL_P["a_backswing"]
        a[ph == 3] = SEL_P["a_followthrough"]
    out = {}
    for jn, k in zip(("shoulder", "elbow", "wrist"), i, strict=True):
        raw = c["xy"][:, k].copy()
        occ = np.isin(c["vis"][f"{arm}_{jn}"], OCCLUDED)
        w = np.clip(np.nan_to_num(c["sc"][:, k]), 0, 1)
        w[w < thr] = 0
        w[np.isnan(raw[:, 0])] = 0
        w[occ] = 0
        p = raw.copy()
        p[w == 0] = np.nan
        if method == "raw":
            r = raw.copy()
        elif method in ("linear", "pchip"):
            r = interp_restore(t, p, w, method)
        else:
            r = smooth_restore(t, p, w, SEL_P["lambda"], a if method == "P" else np.ones(n))
        r = raw.copy() if r is None else r
        res = raw.copy()
        obs = np.where(w > 0)[0]
        if len(obs) >= 2:
            interior = np.zeros(n, bool)
            interior[obs[0] : obs[-1] + 1] = True
            miss = (w == 0) & interior
            res[miss] = r[miss]
            if method in ("U", "P"):
                res[w > 0] = r[w > 0]
        out[k] = res
    return out


def angle_change_metrics(theta, jump_deg=30.0):
    """Frame-to-frame elbow-angle change over a window.

    Only differences between two finite angles count; undefined angles are excluded from both
    numerator and denominator instead of being counted as "no jump".
    Returns (jump_rate, mean_abs_change, n_valid_differences); NaN when nothing is valid.
    """
    d = np.abs(np.diff(np.asarray(theta, dtype=float)))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, 0
    return float((d > jump_deg).mean()), float(d.mean()), int(d.size)
