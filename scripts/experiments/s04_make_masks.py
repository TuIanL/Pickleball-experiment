"""Generate synthetic-occlusion masks for every eligible clip -> outputs/masks_all.csv."""

import _common  # noqa: F401
import pandas as pd

from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import arms, phases
from pickleball_experiment.paths import work
from pickleball_experiment.synthetic import gen_masks

C = load_clips()
rows = []
for sid, c in C.items():
    ph = phases(c)
    if ph is None or not c["final"]["stroke_ok"]:
        continue
    for arm in arms(c):
        for m in gen_masks(c, arm, ph):
            rows.append(
                dict(sid=sid, group=c["group"], origin=c["origin"], stroke=c["stroke"], **m)
            )
M = pd.DataFrame(rows)
M.to_csv(work("masks_all.csv"), index=False)
print("masks", len(M), "clips", M.sid.nunique())
print(M.groupby(["place", "jset", "L"]).agg(n=("sid", "size"), clips=("sid", "nunique")))
