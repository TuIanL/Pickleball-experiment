"""S1: inter-annotator agreement from the two original annotations."""

import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pickleball_experiment.dataset import BOUNDARIES, load_clips

C = load_clips()
rows, va, vb = [], [], []
for sid, c in C.items():
    a, b = c["A"], c["B"]
    r = dict(
        sid=sid,
        arm_agree=a["hitting_arm"] == b["hitting_arm"],
        comp_agree=a["clip_completeness"] == b["clip_completeness"],
        bs_agree=a["backswing_status"] == b["backswing_status"],
    )
    for k in BOUNDARIES:
        x, y = a["boundaries"][k], b["boundaries"][k]
        r[k] = abs(x - y) if (x is not None and y is not None) else np.nan
    rows.append(r)
    for arm in {x for x in (a["hitting_arm"], b["hitting_arm"]) if x in ("left", "right")}:
        for j in ("shoulder", "elbow", "wrist"):
            va += list(c["visA"][f"{arm}_{j}"] != "visible")
            vb += list(c["visB"][f"{arm}_{j}"] != "visible")
D = pd.DataFrame(rows)
for k in BOUNDARIES:
    x = D[k].dropna()
    print(
        k,
        "n",
        len(x),
        "median",
        x.median(),
        "IQR",
        x.quantile(0.25),
        x.quantile(0.75),
        "<=1",
        round((x <= 1).mean(), 3),
        "<=2",
        round((x <= 2).mean(), 3),
    )
print(
    "arm",
    D.arm_agree.mean().round(3),
    "completeness",
    D.comp_agree.mean().round(3),
    "backswing",
    D.bs_agree.mean().round(3),
)
va, vb = np.array(va), np.array(vb)
print(
    "visibility agreement",
    round((va == vb).mean(), 3),
    "kappa",
    round(cohen_kappa_score(va, vb), 3),
)
