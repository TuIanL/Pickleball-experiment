"""E4 (exploratory): four-class stroke recognition from hitting-arm features, leave-one-source-out."""

import _common  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from pickleball_experiment.config import PARAMS, PILOT_GROUP
from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import IDX, arms, elbow_angle, phases, torso_scale
from pickleball_experiment.paths import work
from pickleball_experiment.realocc import restore_clip

C = load_clips()
GRID = np.linspace(0, 1, 32)


def resample(x, tt):
    ok = np.isfinite(x)
    return np.interp(GRID, tt[ok], x[ok]) if ok.sum() >= 2 else np.zeros(32)


def features(c, arm, out):
    i = IDX[arm]
    s, e, w = out[i[0]], out[i[1]], out[i[2]]
    scale = torso_scale(c)
    scale = scale if np.isfinite(scale) else np.nanmedian(np.linalg.norm(s - e, axis=1)) * 2
    rel_w, rel_e = (w - s) / scale, (e - s) / scale
    if arm == "left":
        rel_w[:, 0] *= -1
        rel_e[:, 0] *= -1
    tt = np.linspace(0, 1, c["N"])
    series = [elbow_angle(s, e, w), rel_w[:, 0], rel_w[:, 1], rel_e[:, 0], rel_e[:, 1]]
    feats = []
    for x in (resample(v, tt) for v in series):
        v = np.gradient(x)
        feats += [
            x.mean(),
            x.std(),
            x.min(),
            x.max(),
            x.max() - x.min(),
            np.argmax(x) / 31,
            np.argmin(x) / 31,
            np.abs(v).max(),
            np.abs(v).mean(),
        ]
        feats += list(x[::4])
    feats.append(c["N"] / 30)
    return np.array(feats)


methods = ["raw", "linear", "pchip", "U", "P"]
data = {m: [] for m in methods}
y, grp = [], []
for c in C.values():
    if c["group"] == PILOT_GROUP or not c["final"]["stroke_ok"]:
        continue
    ph, a = phases(c), arms(c)
    if ph is None or not a:
        continue
    arm = a[0] if len(a) == 1 else "right"
    for m in methods:
        data[m].append(features(c, arm, restore_clip(c, arm, ph, m)))
    y.append(c["stroke"])
    grp.append(c["origin"])
y, grp = np.array(y), np.array(grp)
labels = sorted(set(y))
rows = []
for m, feats in data.items():
    X = np.nan_to_num(np.array(feats))
    pred = np.empty(len(y), dtype=object)
    for o in np.unique(grp):
        tr = grp != o
        clf = RandomForestClassifier(
            n_estimators=500, random_state=PARAMS["seed"], class_weight="balanced", n_jobs=-1
        )
        pred[~tr] = clf.fit(X[tr], y[tr]).predict(X[~tr])
    r = dict(input=m, macro_f1=f1_score(y, pred, average="macro"))
    for o in np.unique(grp):
        r[f"f1_{o}"] = f1_score(y[grp == o], pred[grp == o], average="macro")
    for lab, rc in zip(labels, recall_score(y, pred, average=None, labels=labels), strict=True):
        r[f"recall_{lab}"] = rc
    rows.append(r)
T = pd.DataFrame(rows).set_index("input")
T.to_csv(work("e4.csv"))
print("clips", len(y))
print(T.round(3).T)
