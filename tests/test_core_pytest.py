import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from fusion_backbone import FusionBackbone
from drift import MultiViewDriftMonitor
from routing import route_from_signature


def test_fusion_backbone_forward_shapes_and_padding_mask():
    torch.manual_seed(1)
    model = FusionBackbone(encoder_dim=12, gru_hidden=8, cnn_channels=8, fused_dim=10)
    model.eval()
    x = torch.randn(3, 9, 12)
    mask = torch.tensor([[1]*9, [1]*7+[0]*2, [1]*5+[0]*4], dtype=torch.bool)
    out = model(x, mask)
    assert out["sentiment_logits"].shape == (3, 2)
    assert out["fused"].shape == (3, 10)


def test_trias_stable_vs_shifted_and_routing():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(800, 10))
    p = rng.dirichlet([3, 3], size=800)
    m = MultiViewDriftMonitor(n_proj=8, seed=3)
    m.calibrate_from_pool(x, p, window_size=32, n_draws=1100, severe_q=0.995, seed=4)
    ref_x, ref_p = x[:64], p[:64]
    stable = m.score(ref_x, x[64:128], ref_p, p[64:128])
    shifted_x = x[128:192] + 4.0
    shifted_p = np.tile(np.array([[0.98, 0.02]]), (64, 1))
    shifted = m.score(ref_x, shifted_x, ref_p, shifted_p)
    assert stable.severity in {"stable", "mild"}
    assert shifted.severity in {"moderate", "severe"}
    route = route_from_signature(shifted, m)
    assert route.severity == shifted.severity
    assert len(route.components) > 0
