"""E3: real occlusion intervals (adjudicated) treated as missing; plausibility metrics only."""

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import PILOT_GROUP, RESERVED_IDS
from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import IDX, OCCLUDED, arms, elbow_angle, phases
from pickleball_experiment.paths import work
from pickleball_experiment.realocc import restore_clip

C = load_clips()
rows = []
for sid, c in C.items():
    if c["group"] == PILOT_GROUP or sid in RESERVED_IDS or not c["final"]["stroke_ok"]:
        continue
    ph = phases(c)
    if ph is None:
        continue
    n = c["N"]
    f = int(np.where(ph == 2)[0].min())
    cf = int(np.where(ph == 2)[0].max())
    lo, hi = max(0, f - 10), min(n - 1, cf + 10)
    for arm in arms(c):
        i = IDX[arm]
        occ_e = np.isin(c["vis"][f"{arm}_elbow"], OCCLUDED)
        occ_w = np.isin(c["vis"][f"{arm}_wrist"], OCCLUDED)
        occ = occ_e | occ_w
        segs, j = [], 0
        while j < n:
            if occ[j]:
                k = j
                while k + 1 < n and occ[k + 1]:
                    k += 1
                interior = (~occ[:j]).sum() >= 2 and (~occ[k + 1 :]).sum() >= 2
                if k - j + 1 >= 2 and k >= lo and j <= hi and interior:
                    segs.append((j, k))
                j = k + 1
            else:
                j += 1
        vis_all = (~occ_e) & (~occ_w) & (c["vis"][f"{arm}_shoulder"] == "visible")
        if not segs or vis_all.sum() < 3:
            continue
        s0, e0, w0 = (c["xy"][:, k] for k in i)
        mu = np.nanmedian(np.linalg.norm(s0 - e0, axis=1)[vis_all])
        mf = np.nanmedian(np.linalg.norm(w0 - e0, axis=1)[vis_all])
        for meth in ("raw", "linear", "pchip", "U", "P"):
            out = restore_clip(c, arm, ph, meth)
            s, e, w = (out[k] for k in i)
            th = elbow_angle(s, e, w)
            ua, fa = np.linalg.norm(s - e, axis=1), np.linalg.norm(w - e, axis=1)
            for a, b in segs:
                sl = slice(a, b + 1)
                dth = np.abs(np.diff(th[max(a - 1, 0) : min(b + 2, n)]))
                rows.append(
                    dict(
                        sid=sid,
                        origin=c["origin"],
                        meth=meth,
                        len=b - a + 1,
                        limb_dev=float(np.nanmean(np.abs(np.r_[ua[sl] / mu - 1, fa[sl] / mf - 1]))),
                        jump_rate=float((dth > 30).mean()),
                        mean_abs_dtheta=float(np.nanmean(dth)),
                    )
                )
E3 = pd.DataFrame(rows)
E3.to_csv(work("e3.csv"), index=False)
print("segments", (E3.meth == "raw").sum(), "clips", E3.sid.nunique())
cols = ["limb_dev", "jump_rate", "mean_abs_dtheta"]
g = (
    E3.groupby(["meth", "origin", "sid"])[cols]
    .mean()
    .groupby(["meth", "origin"])
    .mean()
    .groupby("meth")
    .mean()
)
print(g.loc[["raw", "linear", "pchip", "U", "P"]].round(3))
