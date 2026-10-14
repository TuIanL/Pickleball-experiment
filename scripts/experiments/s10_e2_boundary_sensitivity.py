"""E2: P with fixed parameters under alternative phase boundaries."""

from multiprocessing import Pool

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import PILOT_GROUP, RESERVED_IDS, SEL_P, SEL_U, is_primary
from pickleball_experiment.dataset import BOUNDARIES, load_clips
from pickleball_experiment.kinematics import mid, phases
from pickleball_experiment.paths import work
from pickleball_experiment.synthetic import run_mask

C = load_clips()
A13 = (SEL_P["a_backswing"], SEL_P["a_followthrough"])


def variants(c):
    v = {"final": phases(c)}
    for who in "AB":
        b = {k: c[who]["boundaries"][k] for k in BOUNDARIES}
        bad = (
            b["forward_start"] is None
            or b["contact_frame"] is None
            or b["contact_frame"] < b["forward_start"]
        )
        v[who] = None if bad else phases(c, bounds=b)
    b0 = {k: mid(c["final"]["bounds"], k) for k in BOUNDARIES}
    for sh in (-2, 2):
        b = {k: (None if x is None else min(max(x + sh, 0), c["N"] - 1)) for k, x in b0.items()}
        v[f"shift{sh:+d}"] = phases(c, bounds=b)
    v["fc_only"] = phases(c, mode="fc")
    ph = v["final"]
    mm = ph.copy()
    mm[ph == 1], mm[ph == 2], mm[ph == 3] = 2, 3, 1
    v["mismatch"] = mm
    return v


def work_one(rec):
    m = pd.Series(rec)
    c = C[m.sid]
    vs = variants(c)
    cond = f"{m.place}_{m.jset}_{m.L}"
    out = []
    for name, ph in vs.items():
        mae = (
            np.nan
            if ph is None
            else run_mask(c, vs["final"], m, "P", SEL_P["lambda"], A13, phase_override=ph)["mae"]
        )
        out.append(dict(sid=m.sid, origin=m.origin, cond=cond, variant=name, mae=mae))
    out.append(
        dict(
            sid=m.sid,
            origin=m.origin,
            cond=cond,
            variant="U",
            mae=run_mask(c, vs["final"], m, "U", SEL_U["lambda"])["mae"],
        )
    )
    return out


if __name__ == "__main__":
    M = pd.read_csv(work("masks_all.csv"))
    M = M[(M.group != PILOT_GROUP) & (~M.sid.isin(RESERVED_IDS))]
    M = M[[is_primary(p, n) for p, n in zip(M.place, M.L, strict=True)]]
    with Pool() as pool:
        res = pool.map(work_one, M.to_dict("records"), chunksize=50)
    R = pd.DataFrame([r for rr in res for r in rr])
    R.to_pickle(work("e2.pkl"))
    ok = R.dropna(subset=["mae"])
    cs = (
        ok.groupby(["variant", "origin", "sid", "cond"])
        .mae.mean()
        .groupby(["variant", "origin", "sid"])
        .mean()
    )
    u = cs.xs("U", level="variant")
    rows = []
    for v in cs.index.get_level_values("variant").unique():
        x = cs.xs(v, level="variant")
        d = (x - u).dropna()
        rows.append(
            dict(
                variant=v,
                n_clips=len(x),
                mae=x.groupby("origin").mean().mean(),
                delta=d.groupby("origin").mean().mean(),
            )
        )
    S = pd.DataFrame(rows)
    S.to_csv(work("e2_summary.csv"), index=False)
    print(S.round(3).to_string(index=False))
