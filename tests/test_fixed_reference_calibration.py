import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from drift import MultiViewDriftMonitor


def test_fixed_reference_calibration_uses_operational_statistic():
    rng=np.random.default_rng(7)
    xref=rng.normal(size=(300,8))
    pref=np.column_stack([np.full(300,.55),np.full(300,.45)])
    xcal=rng.normal(size=(1200,8))
    p1=np.clip(.55+rng.normal(0,.02,1200),.01,.99)
    pcal=np.column_stack([p1,1-p1])
    m=MultiViewDriftMonitor(n_proj=8,seed=3)
    m.calibrate_fixed_reference(xref,pref,xcal,pcal,window_size=64,stride=1,severe_q=.995)
    assert m.thresholds is not None
    assert m.calibration_meta["mode"].startswith("fixed-reference")
    assert m.calibration_meta["comparisons"] == 1137
