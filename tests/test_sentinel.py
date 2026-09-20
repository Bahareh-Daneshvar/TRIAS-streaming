import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from sentinel import make_frozen_sentinel, TrailingWindow


def test_sentinel_is_independent_and_frozen():
    model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,2))
    s=make_frozen_sentinel(model)
    assert s is not model
    assert all(not p.requires_grad for p in s.parameters())
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    assert not torch.equal(next(model.parameters()), next(s.parameters()))


def test_trailing_window_is_fixed_size_and_causal():
    w=TrailingWindow(3)
    w.extend([1,2])
    assert not w.ready()
    w.extend([3,4])
    assert w.ready()
    assert w.values()==[2,3,4]
