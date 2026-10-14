"""Pilot-only calibration of the lambda range and phase multipliers (development clips only)."""

import itertools

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import PILOT_GROUP
from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import phases
from pickleball_experiment.paths import work
from pickleball_experiment.synthetic import run_mask

C = load_clips()
M = pd.read_csv(work("masks_all.csv"))
M = M[M.group == PILOT_GROUP]
wide = 10.0 ** np.arange(-9, -0.5, 0.5)
mult = [0.1, 0.3, 0.5, 1, 2, 3, 10]
rows = []
for _, m in M.iterrows():
    c = C[m.sid]
    ph = phases(c)
    for meth in ("linear", "pchip"):
        rows.append(dict(meth=meth, lam=np.nan, a1=1, a3=1, **run_mask(c, ph, m, meth)))
    for lam in wide:
        rows.append(dict(meth="U", lam=lam, a1=1, a3=1, **run_mask(c, ph, m, "U", lam)))
    for lam in wide[2:-6]:
        for a1, a3 in itertools.product(mult, mult):
            rows.append(
                dict(meth="P", lam=lam, a1=a1, a3=a3, **run_mask(c, ph, m, "P", lam, (a1, a3)))
            )
R = pd.DataFrame(rows)
print(R[R.meth != "P"].groupby(["meth", "lam"], dropna=False).mae.mean().round(3).to_string())
print(R[R.meth == "P"].groupby(["a1", "a3"]).mae.mean().unstack().round(3))
