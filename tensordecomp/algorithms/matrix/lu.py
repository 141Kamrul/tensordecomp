from __future__ import annotations

from typing import Any

import scipy as sc

import numpy as np


def lu(array: np.ndarray) -> dict[str, Any]:
    p, l, u = sc.linalg.lu(array)
    return {"l": l, "u": u}