"""Frozen parameters (configs/frozen_params.json) and the reserved reference subset."""

import json

import numpy as np

from .paths import CONFIG_DIR

PARAMS = json.loads((CONFIG_DIR / "frozen_params.json").read_text(encoding="utf-8"))
LAMBDA_GRID = 10.0 ** np.array(PARAMS["lambda_grid_log10"])
PHASE_MULTIPLIERS = PARAMS["phase_multipliers"]
SEL_U = PARAMS["selected_by_loso"]["U"]
SEL_P = PARAMS["selected_by_loso"]["P"]
RESERVED_IDS = set(
    json.loads((CONFIG_DIR / "reference_subset_ids.json").read_text(encoding="utf-8"))["H_ids"]
)
PILOT_GROUP = "development_pilot_48"


def is_primary(place, length):
    """The 9-frame contact-centred condition is a stress test, not part of the primary analysis."""
    return not (place == "contact" and length == 9)
