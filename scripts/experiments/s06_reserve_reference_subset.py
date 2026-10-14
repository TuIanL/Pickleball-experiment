"""Reserve 24 clips (2 per source x stroke) before the formal run; regenerate the frame list."""

import json
import zlib

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.config import PILOT_GROUP
from pickleball_experiment.paths import CONFIG_DIR, work

M = pd.read_csv(work("masks_all.csv"))
elig = (
    M[M.group != PILOT_GROUP]
    .groupby("sid")
    .agg(origin=("origin", "first"), stroke=("stroke", "first"))
)
has3 = M[(M.place == "random") & (M.L == 3)].sid.unique()
elig = elig[elig.index.isin(has3)].reset_index().sort_values("sid")
rng = np.random.default_rng(zlib.crc32(b"20260907-human-reference-v1"))
ids = []
for _, g in elig.groupby(["origin", "stroke"]):
    ids += sorted(rng.choice(g.sid.values, size=min(2, len(g)), replace=False).tolist())
ids = sorted(ids)
out = CONFIG_DIR / "reference_subset_ids.json"
saved = json.loads(out.read_text(encoding="utf-8"))["H_ids"]
print("reserved", len(ids), "matches committed list:", ids == saved)
