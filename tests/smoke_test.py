import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from fusion_backbone import FusionBackbone
from drift import MultiViewDriftMonitor
from adaptation import set_trainable
from trias import TRIASController


def main():
    torch.manual_seed(0)
    model = FusionBackbone(encoder_dim=32, gru_hidden=16, gru_layers=2, cnn_channels=8, fused_dim=32)
    x = torch.randn(12, 20, 32)
    mask = torch.ones(12, 20, dtype=torch.bool)
    out = model(x, mask)
    assert out["fused"].shape == (12, 32)
    assert out["sentiment_logits"].shape == (12, 2)
    assert out["risk_level_logits"].shape == (12, 3)

    rng = np.random.default_rng(0)
    ref = rng.normal(size=(256, 16))
    p_ref = rng.dirichlet([3, 2], size=256)

    mon = MultiViewDriftMonitor(n_proj=16, seed=0)
    raw = []
    for _ in range(1200):
        cur = rng.normal(scale=1.02, size=(128, 16))
        p = rng.dirichlet([3, 2], size=128)
        raw.append(mon.raw(ref, cur, p_ref, p))
    mon.calibrate(np.vstack(raw), mild_q=0.90, moderate_q=0.97, severe_q=0.995)
    trias = TRIASController(mon)

    # Near-stationary stream should usually be stable or, by calibrated design, rarely mild+.
    stable_decision = trias.decide(
        ref,
        rng.normal(scale=1.01, size=(128, 16)),
        p_ref,
        rng.dirichlet([3, 2], size=128),
    )

    # Strong induced representation + predictive shift should escalate.
    shifted = rng.normal(loc=1.2, scale=1.25, size=(128, 16))
    p_shift = rng.dirichlet([1, 6], size=128)
    decision = trias.decide(ref, shifted, p_ref, p_shift)
    assert decision.scores.composite >= 0
    assert decision.scores.severity in {"moderate", "severe"}

    comps = set_trainable(model, decision.scores.severity)
    trainable = sum(p.requires_grad for p in model.parameters())

    print("Fusion host forward pass: OK")
    print("TRIAS thresholds:", mon.thresholds)
    print("Near-stationary severity:", stable_decision.scores.severity)
    print("Shifted severity:", decision.scores.severity)
    print("Shifted score:", decision.scores.composite)
    print("Adaptation components:", comps)
    print("Trainable parameter tensors:", trainable)


if __name__ == "__main__":
    main()
