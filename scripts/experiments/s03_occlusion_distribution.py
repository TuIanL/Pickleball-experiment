"""E0: occlusion by phase and joint; keypoint score vs human-labelled occlusion (ROC-AUC)."""

import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import IDX, OCCLUDED, arms, phases
from pickleball_experiment.paths import work

C = load_clips()
rows, ys, ss = [], [], []
for sid, c in C.items():
    ph = phases(c)
    if ph is None:
        continue
    for arm in arms(c):
        for jn, k in zip(("shoulder", "elbow", "wrist"), IDX[arm], strict=True):
            v = c["vis"][f"{arm}_{jn}"]
            for p in (1, 2, 3):
                m = ph == p
                if m.sum():
                    occ = np.isin(v[m], OCCLUDED).mean()
                    rows.append(
                        dict(
                            sid=sid,
                            group=c["group"],
                            stroke=c["stroke"],
                            joint=jn,
                            phase=p,
                            occ=occ,
                        )
                    )
            sc = c["sc"][:, k]
            good = ~np.isnan(sc) & np.isin(v, ["visible", *OCCLUDED])
            ys += list(np.isin(v[good], OCCLUDED).astype(int))
            ss += list(-sc[good])
D = pd.DataFrame(rows)
D.to_csv(work("e0_occ.csv"), index=False)
F = D[D.group != "development_pilot_48"]
print((F.groupby(["joint", "phase"]).occ.mean().unstack() * 100).round(1))
print((F[F.joint != "shoulder"].groupby(["stroke", "phase"]).occ.mean().unstack() * 100).round(1))
print("ROC-AUC (low score -> occluded)", round(roc_auc_score(ys, ss), 3), "frames", len(ys))
