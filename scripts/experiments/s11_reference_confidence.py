"""Sensitivity of the P-U difference to the confidence of the reference keypoints."""

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import SEL_P, SEL_U, is_primary
from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import IDX
from pickleball_experiment.paths import work

C = load_clips()
R = pd.read_pickle(work("e1_S.pkl"))
keep = ((R.meth == "U") & np.isclose(R.lam, SEL_U["lambda"])) | (
    (R.meth == "P")
    & np.isclose(R.lam, SEL_P["lambda"])
    & (R.a1 == SEL_P["a_backswing"])
    & (R.a3 == SEL_P["a_followthrough"])
)
X = R[keep & np.array([is_primary(p, n) for p, n in zip(R.place, R.L, strict=True)])].copy()
key = X[["sid", "arm", "start", "L"]].drop_duplicates()
key["ms"] = [
    float(np.nanmin(C[r.sid]["sc"][r.start : r.start + r.L, list(IDX[r.arm])]))
    for r in key.itertuples()
]
X = X.merge(key, on=["sid", "arm", "start", "L"])
for thr in (0.3, 0.5, 0.7):
    for label, Y in (("all", X[X.ms >= thr]), ("forward/contact", X[(X.ms >= thr) & (X.cp == 2)])):
        cs = (
            Y.groupby(["meth", "origin", "sid", "place", "jset", "L"])
            .mae.mean()
            .groupby(["meth", "origin", "sid"])
            .mean()
        )
        d = (cs.xs("P", level="meth") - cs.xs("U", level="meth")).dropna()
        print(
            f">= {thr} {label:16s} clips {d.size:4d}  P-U {d.groupby('origin').mean().mean():+.3f}"
        )
