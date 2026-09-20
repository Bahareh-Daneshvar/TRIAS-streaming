import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from reference import ConservativeReferenceManager

rng = np.random.default_rng(2)
x = rng.normal(size=(100, 4)); p = rng.dirichlet([2,2], size=100)
m = ConservativeReferenceManager(x, p, max_samples=120, stable_patience=2)
assert not m.observe(rng.normal(size=(20,4)), rng.dirichlet([2,2], size=20), "stable")
assert m.observe(rng.normal(size=(20,4)), rng.dirichlet([2,2], size=20), "stable")
assert m.state.updates == 1
assert len(m.state.x_ref) == 120
assert not m.observe(rng.normal(size=(20,4)), rng.dirichlet([2,2], size=20), "severe")
print("Reference manager: OK")
