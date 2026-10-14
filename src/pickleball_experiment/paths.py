"""Local paths. Data stay outside Git; set PB_DATA_ROOT / PB_WORK_DIR to override."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Root of the WRSSC2026_complete_947_20261004 package (videos/, skeleton/, annotations/, ...)
DATA_ROOT = Path(
    os.environ.get("PB_DATA_ROOT", REPO_ROOT / "data" / "WRSSC2026_complete_947_20261004")
)
# Intermediate and result files (ignored by Git)
WORK_DIR = Path(os.environ.get("PB_WORK_DIR", REPO_ROOT / "outputs"))
CONFIG_DIR = REPO_ROOT / "configs"


def work(name: str) -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR / name
