"""E1: leave-one-source-out selection, per-source / equal-weight MAE, paired delta and bootstrap CI."""

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import PARAMS, is_primary
from pickleball_experiment.paths import work

R = pd.read_pickle(work("e1_S.pkl"))
R["cond"] = R.place + "_" + R.jset + "_L" + R.L.astype(str)
R["cfg"] = [
    f"{m}|{lam:g}|{a:g}|{b:g}"
    for m, lam, a, b in zip(R.meth, R.lam.fillna(-1), R.a1.fillna(-1), R.a3.fillna(-1), strict=True)
]
P = R[[is_primary(p, n) for p, n in zip(R.place, R.L, strict=True)]].dropna(subset=["mae"])
# mask -> condition mean within clip -> clip mean (equal weight over conditions)
cs = P.groupby(["cfg", "origin", "sid", "cond"]).mae.mean().groupby(["cfg", "origin", "sid"]).mean()
origins = sorted(R.origin.unique())


def source_equal(g):
    return g.groupby(["cfg", "origin"]).mean().groupby("cfg").mean()


sel, rows = {}, []
for o in origins:
    train = source_equal(cs[cs.index.get_level_values("origin") != o])
    test = cs[cs.index.get_level_values("origin") == o]
    for meth in ("linear", "pchip", "U", "P"):
        best = train[[k for k in train.index if k.startswith(meth + "|")]].idxmin()
        sel[(o, meth)] = best
        v = test.xs(best, level="cfg")
        rows.append(dict(origin=o, meth=meth, cfg=best, mae=v.mean(), n=len(v)))
    u_test = test[test.index.get_level_values("cfg").str.startswith("U|")].groupby("cfg").mean()
    rows.append(dict(origin=o, meth="U_oracle", cfg=u_test.idxmin(), mae=u_test.min(), n=np.nan))
T = pd.DataFrame(rows)
tab = T.pivot(index="meth", columns="origin", values="mae")
tab["equal_weight"] = tab.mean(axis=1)
print(T.to_string())
print(tab.round(3))


def per_clip(meth):
    parts = []
    for o in origins:
        v = cs.xs(sel[(o, meth)], level="cfg")
        parts.append(v[v.index.get_level_values("origin") == o])
    return pd.concat(parts)


d = (per_clip("P") - per_clip("U")).dropna()
print(
    "delta P-U equal-weight",
    round(d.groupby("origin").mean().mean(), 4),
    "per source",
    d.groupby("origin").mean().round(3).to_dict(),
    "median",
    round(d.median(), 4),
    "P better",
    round((d < 0).mean(), 3),
)
rng = np.random.default_rng(PARAMS["seed"])
groups = [d.xs(o, level="origin").values for o in origins]
bs = [
    np.mean([rng.choice(v, len(v)).mean() for v in groups])
    for _ in range(PARAMS["bootstrap_iterations"])
]
ci = np.percentile(bs, [2.5, 97.5])
print("95% CI", ci.round(4))
pd.to_pickle(dict(sel=sel, cs=cs, T=T, d=d), work("e1_summary.pkl"))
summary = tab["equal_weight"].rename("mae_equal_weight").to_frame()
summary.loc["delta_P_minus_U", "mae_equal_weight"] = d.groupby("origin").mean().mean()
summary.loc["delta_ci_low", "mae_equal_weight"] = ci[0]
summary.loc["delta_ci_high", "mae_equal_weight"] = ci[1]
summary.to_csv(work("e1_table.csv"))
