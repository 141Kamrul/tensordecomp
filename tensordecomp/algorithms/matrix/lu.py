from __future__ import annotations

from typing import Any


import numpy as np


def lu(array: np.ndarray) -> dict[str, Any]:
    p, l, u = np.linalg.lu(array)
    return {"l": l, "u": u}