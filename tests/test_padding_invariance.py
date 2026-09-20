import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from fusion_backbone import FusionBackbone

torch.manual_seed(4)
model = FusionBackbone(encoder_dim=8, gru_hidden=6, gru_layers=1, cnn_channels=5, fused_dim=12, dropout=0.0)
model.eval()

# Same 7-token sequence, once alone and once followed by arbitrary masked padding.
base = torch.randn(1, 7, 8)
padded = torch.cat([base, torch.randn(1, 5, 8) * 100], dim=1)
mask_base = torch.ones(1, 7, dtype=torch.bool)
mask_pad = torch.tensor([[1]*7 + [0]*5], dtype=torch.bool)
with torch.no_grad():
    a = model(base, mask_base)['fused']
    b = model(padded, mask_pad)['fused']
assert torch.allclose(a, b, atol=1e-5), (a-b).abs().max().item()
print('Padding invariance: OK')
