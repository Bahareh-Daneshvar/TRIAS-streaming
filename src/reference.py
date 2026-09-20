from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ReferenceState:
    x_ref: np.ndarray
    p_ref: np.ndarray
    stable_streak: int = 0
    updates: int = 0


class ConservativeReferenceManager:
    """Optional reference maintenance control, not a claimed novelty.

    The reference is updated only after several consecutive stable chunks. This
    avoids immediately absorbing a newly detected shift into the baseline.
    """

    def __init__(self, x_ref, p_ref, max_samples=1024, stable_patience=3):
        x_ref = np.asarray(x_ref, dtype=float)
        p_ref = np.asarray(p_ref, dtype=float)
        if x_ref.ndim != 2 or p_ref.ndim != 2 or len(x_ref) != len(p_ref):
            raise ValueError("Reference arrays must be aligned 2D arrays")
        self.max_samples = int(max_samples)
        self.stable_patience = int(stable_patience)
        if self.max_samples < 16 or self.stable_patience < 1:
            raise ValueError("Invalid reference manager settings")
        self.state = ReferenceState(x_ref[-self.max_samples:].copy(), p_ref[-self.max_samples:].copy())

    def current(self):
        return self.state.x_ref, self.state.p_ref

    def observe(self, x_cur, p_cur, severity: str):
        if severity != "stable":
            self.state.stable_streak = 0
            return False
        self.state.stable_streak += 1
        if self.state.stable_streak < self.stable_patience:
            return False
        x = np.asarray(x_cur, dtype=float)
        p = np.asarray(p_cur, dtype=float)
        self.state.x_ref = np.concatenate([self.state.x_ref, x], axis=0)[-self.max_samples:]
        self.state.p_ref = np.concatenate([self.state.p_ref, p], axis=0)[-self.max_samples:]
        self.state.updates += 1
        self.state.stable_streak = 0
        return True
