"""Trajectory restoration: linear / PCHIP interpolation and U / P second-order smoothing (MAP form).

L(q) = 1/2 sum_t w_t ||q_t - p_t||^2 + lambda/2 sum_t rho_t a_t ||D2 q_t||^2
U uses a_t = 1; P uses a phase-dependent multiplier (forward swing/contact fixed to 1).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

_D2_CACHE = {}


def d2_matrix(t):
    """Second difference on a non-uniform grid and its time-integration weights rho.

    Cached on the exact bytes of the time axis, so two clips share a matrix only if every
    timestamp is identical.
    """
    t = np.ascontiguousarray(t, dtype=float)
    key = t.tobytes()
    if key not in _D2_CACHE:
        n = len(t)
        d = np.zeros((max(n - 2, 0), n))
        rho = np.zeros(max(n - 2, 0))
        for i in range(1, n - 1):
            hm, hp = t[i] - t[i - 1], t[i + 1] - t[i]
            k = 2 / (hm + hp)
            d[i - 1, i - 1] = k / hm
            d[i - 1, i] = -k * (1 / hm + 1 / hp)
            d[i - 1, i + 1] = k / hp
            rho[i - 1] = (hm + hp) / 2
        _D2_CACHE[key] = (d, rho)
    return _D2_CACHE[key]


def smooth_restore(t, p, w, lam, a):
    """t (N,), p (N,2) with NaN where missing, w (N,) weights (0 = missing), a (N,) multipliers."""
    if (w > 0).sum() < 2:
        return None
    d, rho = d2_matrix(t)
    q = lam * (d.T * (rho * a[1:-1])) @ d
    try:
        return np.linalg.solve(np.diag(w) + q, w[:, None] * np.nan_to_num(p))
    except np.linalg.LinAlgError:
        return None


def interp_restore(t, p, w, kind):
    """Per-coordinate 'linear' or 'pchip' interpolation in time, no extrapolation."""
    obs = w > 0
    if obs.sum() < 2:
        return None
    out = np.full_like(p, np.nan)
    to = t[obs]
    for dim in range(2):
        if kind == "linear":
            v = np.interp(t, to, p[obs, dim])
            v[(t < to[0]) | (t > to[-1])] = np.nan
        else:
            v = PchipInterpolator(to, p[obs, dim], extrapolate=False)(t)
        out[:, dim] = v
    out[obs] = p[obs]
    return out
