"""E1 stratified description with the LOSO-selected configurations."""

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import SEL_P, SEL_U, is_primary
from pickleball_experiment.paths import work

R = pd.read_pickle(work("e1_S.pkl"))
keep = (
    R.meth.isin(["linear", "pchip"])
    | ((R.meth == "U") & np.isclose(R.lam, SEL_U["lambda"]))
    | (
        (R.meth == "P")
        & np.isclose(R.lam, SEL_P["lambda"])
        & (R.a1 == SEL_P["a_backswing"])
        & (R.a3 == SEL_P["a_followthrough"])
    )
)
X = R[keep]
primary = X[[is_primary(p, n) for p, n in zip(X.place, X.L, strict=True)]]


def table(df, by):
    g = (
        df.groupby([*by, "meth", "origin", "sid"])
        .mae.mean()
        .groupby([*by, "meth", "origin"])
        .mean()
        .groupby([*by, "meth"])
        .mean()
        .unstack("meth")
    )
    t = g[["linear", "pchip", "U", "P"]].join(
        df[df.meth == "P"].groupby(by).sid.nunique().rename("n")
    )
    t["P-U"] = t.P - t.U
    return t.round(2)


print(table(primary, ["cp"]))
print(table(X, ["place", "L"]))
print(table(primary, ["jset"]))
print(table(primary, ["cross"]))
print(table(primary, ["stroke"]))
print(primary.groupby("meth").kp.mean().round(4))
