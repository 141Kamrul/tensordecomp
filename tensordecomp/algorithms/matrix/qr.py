from __future__ import annotations

from typing import Any

import numpy as np


def qr(array: np.ndarray) -> dict[str, Any]:
    q, r = np.linalg.qr(array)
    return {"q": q, "r": r}