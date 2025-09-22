from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass
class Signal:
    """Generic signal container."""
    x: np.ndarray                 # samples (complex or real)
    fs: float                     # sample rate [Hz]
    unit: str = "a.u."            # "W" (power), "V", "A", or "a.u."
    meta: Dict = field(default_factory=dict)

    def copy(self) -> "Signal":
        return Signal(self.x.copy(), self.fs, self.unit, self.meta.copy())
