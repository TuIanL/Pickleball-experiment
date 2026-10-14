"""S0: per-clip inclusion fields and descriptive checks -> outputs/s0_table.csv."""

import _common  # noqa: F401
import numpy as np
import pandas as pd

from pickleball_experiment.dataset import load_clips
from pickleball_experiment.kinematics import IDX, mid
from pickleball_experiment.paths import work

C = load_clips()
rows = []
for sid, c in C.items():
    f = c["final"]
    b = f["bounds"]
    arm = f["hitting_arm"]
    r = dict(
        sid=sid,
        group=c["group"],
        origin=c["origin"],
        stroke=c["stroke"],
        N=c["N"],
        arm=arm,
        src=f["source"],
        target_ok=f["target_ok"],
        stroke_ok=f["stroke_ok"],
        comp=f["clip_completeness"],
        bs=f["backswing_status"],
        side=c["side"],
        b=mid(b, "backswing_start"),
        f=mid(b, "forward_start"),
        c=mid(b, "contact_frame"),
        e=mid(b, "followthrough_end"),
    )
    if arm in ("left", "right"):
        ix = list(IDX[arm])
        r["pose_missing_frac"] = np.isnan(c["xy"][:, ix, 0]).any(1).mean()
        r["score_mean"] = np.nanmean(c["sc"][:, ix])
        vis = [c["vis"][f"{arm}_{j}"] == "visible" for j in ("shoulder", "elbow", "wrist")]
        r["allvis_frac"] = np.all(vis, 0).mean()
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(work("s0_table.csv"), index=False)
print(D.group.value_counts().to_dict())
print("arm", D.arm.value_counts().to_dict(), "| stroke unconfirmed", (D.stroke_ok != True).sum())  # noqa: E712
print("f missing", D.f.isna().sum(), "| c missing", D.c.isna().sum())
print("forward/contact length (frames):", (D.c - D.f + 1).describe().round(2).to_dict())
