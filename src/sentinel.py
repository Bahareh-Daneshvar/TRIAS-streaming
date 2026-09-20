from __future__ import annotations
import copy
from collections import deque
import torch


def make_frozen_sentinel(model: torch.nn.Module, device=None) -> torch.nn.Module:
    """Deep-copy the pre-stream model for adaptation-independent monitoring.

    Detector features must live in a fixed representation space. If drift
    features are extracted from the same model that is being adapted, parameter
    updates can masquerade as distribution shift. The sentinel is never trained.
    """
    sentinel = copy.deepcopy(model)
    if device is not None:
        sentinel = sentinel.to(device)
    sentinel.eval()
    for p in sentinel.parameters():
        p.requires_grad = False
    return sentinel


class TrailingWindow:
    """Fixed-size FIFO monitor window populated only with already-seen rows."""
    def __init__(self, size: int):
        if size < 2:
            raise ValueError("size must be >=2")
        self.size = int(size)
        self._items = deque(maxlen=self.size)

    def extend(self, items):
        self._items.extend(items)

    def ready(self) -> bool:
        return len(self._items) == self.size

    def values(self):
        if not self.ready():
            raise RuntimeError("monitor window is not full yet")
        return list(self._items)

    def __len__(self):
        return len(self._items)
