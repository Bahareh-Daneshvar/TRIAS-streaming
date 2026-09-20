import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from drift import MultiViewDriftMonitor, DriftScores
from routing import route_from_signature

# Minimal calibrated monitor state for routing unit tests.
m = MultiViewDriftMonitor()
m.center = np.zeros(3); m.scale = np.ones(3)
m.t_mild, m.t_moderate, m.t_severe = 1.0, 2.0, 3.0
m.z_high = np.array([2.5, 2.5, 2.5])

out_only = DriftScores(0,0,0, 0.3,5.0,0.2, 4.0, 'severe')
r = route_from_signature(out_only, m)
assert r.components == ('gate','heads')
assert 'encoder_top' not in r.components

rep = DriftScores(0,0,0, 4.0,1.0,1.2, 4.0, 'severe')
r = route_from_signature(rep, m)
assert 'encoder_top' in r.components

dep = DriftScores(0,0,0, 0.5,0.8,4.0, 3.5, 'severe')
r = route_from_signature(dep, m)
assert 'encoder_top' not in r.components
assert 'bigru' in r.components and 'textcnn' in r.components
print('Signature routing: OK')
