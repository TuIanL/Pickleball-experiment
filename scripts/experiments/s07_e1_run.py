"""E1: run every method/configuration on every formal mask -> outputs/e1_S.pkl.

Formal set excludes the development pilot and the reserved reference subset.
Usage: python scripts/experiments/s07_e1_run.py [S|R]   (R = reserved subset only)
"""

import itertools
import sys
from multiprocessing import Pool

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import LAMBDA_GRID, PHASE_MULTIPLIERS, PILOT_GROUP, RESERVED_IDS
from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import phases
from pickleball_experiment.paths import work
from pickleball_experiment.synthetic import run_mask

C = load_clips()


def work_one(rec):
    m = pd.Series(rec)
    c = C[m.sid]
    ph = phases(c)
    base = dict(
        sid=m.sid,
        origin=m.origin,
        stroke=m.stroke,
        arm=m.arm,
        jset=m.jset,
        L=m.L,
        place=m.place,
        rep=m.rep,
        start=m.start,
        cp=m.center_phase,
        within=m.within_phase,
        cross=m.crosses_contact,
    )
    out = []
    for meth in ("linear", "pchip"):
        out.append(
            dict(base, meth=meth, lam=np.nan, a1=np.nan, a3=np.nan, **run_mask(c, ph, m, meth))
        )
    for lam in LAMBDA_GRID:
        out.append(dict(base, meth="U", lam=lam, a1=1, a3=1, **run_mask(c, ph, m, "U", lam)))
        for a1, a3 in itertools.product(PHASE_MULTIPLIERS, PHASE_MULTIPLIERS):
            res = run_mask(c, ph, m, "P", lam, (a1, a3))
            out.append(dict(base, meth="P", lam=lam, a1=a1, a3=a3, **res))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "S"
    M = pd.read_csv(work("masks_all.csv"))
    if which == "S":
        M = M[(M.group != PILOT_GROUP) & (~M.sid.isin(RESERVED_IDS))]
    else:
        M = M[M.sid.isin(RESERVED_IDS)]
    print(which, "masks", len(M), "clips", M.sid.nunique(), flush=True)
    with Pool() as pool:
        res = pool.map(work_one, M.to_dict("records"), chunksize=50)
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_pickle(work(f"e1_{which}.pkl"))
    print("rows", len(R))
