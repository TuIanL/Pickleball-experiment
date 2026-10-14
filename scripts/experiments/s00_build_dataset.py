"""Build outputs/clips.pkl from the hand-off package at PB_DATA_ROOT."""

import pickle

import _common  # noqa: F401

from pickleball_experiment.dataset import build_clips
from pickleball_experiment.paths import DATA_ROOT, work

if __name__ == "__main__":
    clips = build_clips(DATA_ROOT)
    with open(work("clips.pkl"), "wb") as fh:
        pickle.dump(clips, fh)
    print("clips", len(clips), "->", work("clips.pkl"))
