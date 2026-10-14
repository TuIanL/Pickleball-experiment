"""Synthetic checks of angle computation, restoration and phase labelling (no real data)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pickleball_experiment.kinematics import elbow_angle, phases  # noqa: E402
from pickleball_experiment.restoration import interp_restore, smooth_restore  # noqa: E402


def test_elbow_angle_right_and_straight():
    s, e = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    assert np.isclose(elbow_angle(s, e, np.array([[1.0, 1.0]]))[0], 90.0)
    assert np.isclose(elbow_angle(s, e, np.array([[2.0, 0.0]]))[0], 180.0)


def test_smoothing_recovers_linear_motion_in_gap():
    t = np.arange(20) / 30.0
    truth = np.stack([3 * t + 1, -2 * t], 1)
    w = np.ones(20)
    w[8:12] = 0
    p = truth.copy()
    p[8:12] = np.nan
    for a in (np.ones(20), np.where(np.arange(20) < 10, 2.0, 0.5)):
        q = smooth_restore(t, p, w, 1e-4, a)
        assert np.allclose(q, truth, atol=1e-8)  # straight lines lie in the null space of D2


def test_interpolation_does_not_extrapolate():
    t = np.arange(6) / 30.0
    p = np.stack([t, t], 1)
    w = np.array([0, 1, 1, 0, 1, 0], float)
    out = interp_restore(t, np.where(w[:, None] > 0, p, np.nan), w, "linear")
    assert np.isnan(out[0]).all() and np.isnan(out[5]).all()
    assert np.allclose(out[3], p[3])


def test_phase_labels():
    c = {
        "N": 12,
        "final": {
            "bounds": {
                "backswing_start": {"low": 1, "high": 1},
                "forward_start": {"low": 4, "high": 5},
                "contact_frame": {"low": 6, "high": 6},
                "followthrough_end": {"low": 9, "high": 9},
            }
        },
    }
    ph = phases(c)
    assert ph.tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0]
