import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from drift import DriftScores
from pilot_b0_b7 import POLICIES, policy_components


class Monitor: z_high = np.array([2., 2., 2.])


def score(severity="moderate", zr=3., zo=1., zd=1.):
    return DriftScores(0, 0, 0, zr, zo, zd, 3., severity)


def test_registry_and_core_actions():
    assert len(POLICIES) == 9 and POLICIES[0] == "B0_static" and POLICIES[7] == "B7_trias"
    shuffled = [("heads",)]
    assert policy_components("B0_static", 0, score(), Monitor(), 0., 1., shuffled) == tuple()
    assert policy_components("B3_error_triggered", 0, score(), Monitor(), .4, .5, shuffled)
    assert policy_components("B4_representation_only", 0, score(zr=1.), Monitor(), 1., .5, shuffled) == tuple()
    assert policy_components("A9_route_shuffle", 0, score(), Monitor(), 1., .5, shuffled) == ("heads",)
