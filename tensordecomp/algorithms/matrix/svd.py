from __future__ import annotations

from typing import Any

import numpy as np


def svd(array: np.ndarray) -> dict[str, Any]:
    u, singular_values, vh = np.linalg.svd(array, full_matrices=False)
    return {"u": u, "singular_values": singular_values, "vh": vh}