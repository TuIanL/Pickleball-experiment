"""Synthetic occlusion masks and per-mask evaluation (2D elbow-angle MAE)."""

import zlib

import numpy as np

from .kinematics import IDX, elbow_angle, ref_ok, torso_scale
from .restoration import interp_restore, smooth_restore

SEED = 20260907


def observed(c, k, thr=0.3):
    return (~np.isnan(c["xy"][:, k, 0])) & (np.nan_to_num(c["sc"][:, k]) >= thr)


def gen_masks(c, arm, ph, lengths_rand=(2, 3, 6), lengths_cc=(3, 6, 9), win=10):
    n = c["N"]
    i = IDX[arm]
    ok = ref_ok(c, arm)
    f = int(np.where(ph == 2)[0].min())
    cf = int(np.where(ph == 2)[0].max())
    out = []
    for jset in ("W", "EW"):
        ks = [i[2]] if jset == "W" else [i[1], i[2]]
        obs = [observed(c, k) for k in ks]

        def admissible(s, length, obs=obs):
            e = s + length - 1
            if s < 0 or e > n - 1 or not ok[s : e + 1].all():
                return False
            return all(o[:s].sum() >= 2 and o[e + 1 :].sum() >= 2 for o in obs)

        for length in lengths_rand:
            starts = sorted(
                {
                    ctr - (length - 1) // 2
                    for ctr in range(f - win, cf + win + 1)
                    if admissible(ctr - (length - 1) // 2, length)
                }
            )
            if starts:
                key = f"{c['sid']}|{arm}|{jset}|{length}".encode()
                rng = np.random.default_rng([SEED, zlib.crc32(key)])
                pick = rng.choice(starts, size=min(3, len(starts)), replace=False)
                for k, s in enumerate(sorted(int(x) for x in pick)):
                    out.append(dict(arm=arm, jset=jset, L=length, place="random", rep=k, start=s))
        for length in lengths_cc:
            s = cf - (length - 1) // 2
            if admissible(s, length):
                out.append(dict(arm=arm, jset=jset, L=length, place="contact", rep=0, start=s))
    for m in out:
        e = m["start"] + m["L"] - 1
        ctr = (m["start"] + e) // 2
        m["center_phase"] = int(ph[ctr])
        m["within_phase"] = bool((ph[m["start"] : e + 1] == ph[ctr]).all())
        m["crosses_contact"] = bool(m["start"] <= cf < e) or bool(m["start"] < cf + 1 <= e)
    return out


def run_mask(c, ph, m, method, lam=None, a13=(1, 1), thr=0.3, phase_override=None):
    """Delete the masked joint(s), restore with `method`, return MAE vs the deleted pose estimate."""
    i = IDX[m["arm"]]
    s, e = int(m["start"]), int(m["start"]) + int(m["L"]) - 1
    ks = [i[2]] if m["jset"] == "W" else [i[1], i[2]]
    t, n = c["t"], c["N"]
    pha = phase_override if phase_override is not None else ph
    a = np.ones(n)
    a[pha == 1] = a13[0]
    a[pha == 3] = a13[1]
    rest = {}
    for k in ks:
        p = c["xy"][:, k].copy()
        w = np.clip(np.nan_to_num(c["sc"][:, k]), 0, 1)
        w[w < thr] = 0
        w[np.isnan(p[:, 0])] = 0
        w[s : e + 1] = 0
        p[s : e + 1] = np.nan
        if method == "raw":
            out = None
        elif method in ("linear", "pchip"):
            out = interp_restore(t, p, w, method)
        else:
            out = smooth_restore(t, p, w, lam, a if method == "P" else np.ones(n))
        rest[k] = out
    sh, el, wr = (c["xy"][s : e + 1, j] for j in i)
    ref = elbow_angle(sh, el, wr)
    if any(v is None for v in rest.values()):
        return dict(mae=np.nan, cov=0.0, kp=np.nan)
    el2 = rest[i[1]][s : e + 1] if i[1] in rest else el
    est = elbow_angle(sh, el2, rest[i[2]][s : e + 1])
    good = np.isfinite(est)
    if good.sum() == 0:
        return dict(mae=np.nan, cov=0.0, kp=np.nan)
    kp = np.mean(
        [
            np.linalg.norm(rest[k][s : e + 1] - c["xy"][s : e + 1, k], axis=1)[good].mean()
            for k in rest
        ]
    ) / torso_scale(c)
    return dict(mae=float(np.abs(est - ref)[good].mean()), cov=float(good.mean()), kp=float(kp))
