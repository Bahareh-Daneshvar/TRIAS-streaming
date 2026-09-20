from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SyntheticStream:
    representations: list[np.ndarray]
    probabilities: list[np.ndarray]
    known_shift_chunks: tuple[int, ...]
    name: str


def _probs(rng, n, strength=0.0):
    # Binary predictive probabilities with controllable prior/confidence shift.
    a = max(0.5, 3.0 - 1.8 * strength)
    b = max(0.5, 2.0 + 3.5 * strength)
    return rng.dirichlet([a, b], size=n)


def make_stream(kind="abrupt", n_chunks=90, chunk_size=96, dim=12, seed=7):
    rng = np.random.default_rng(seed)
    reps, probs = [], []
    onsets = ()
    for t in range(n_chunks):
        shift = 0.0
        scale = 1.0
        dep = 0.0
        if kind == "no_shift":
            pass
        elif kind == "abrupt":
            onsets = (35,)
            if t >= onsets[0]:
                shift, scale, dep = 0.85, 1.15, 0.45
        elif kind == "gradual":
            onsets = (30,)
            if 30 <= t < 50:
                r = (t - 30) / 20.0
                shift, scale, dep = 0.85 * r, 1.0 + 0.15 * r, 0.45 * r
            elif t >= 50:
                shift, scale, dep = 0.85, 1.15, 0.45
        elif kind == "recurring":
            onsets = (25, 65)
            if 25 <= t < 45 or t >= 65:
                shift, scale, dep = 0.85, 1.15, 0.45
        else:
            raise ValueError(f"Unknown stream kind: {kind}")

        x = rng.normal(loc=shift, scale=scale, size=(chunk_size, dim))
        if dep > 0:
            # Induce dependency change in the first feature block.
            shared = rng.normal(size=(chunk_size, 1))
            x[:, : min(4, dim)] += dep * shared
        reps.append(x)
        probs.append(_probs(rng, chunk_size, strength=min(1.0, shift / 0.85 if shift else 0.0)))
    return SyntheticStream(reps, probs, tuple(onsets), kind)
