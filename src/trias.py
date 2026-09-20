from __future__ import annotations
from dataclasses import dataclass
from drift import MultiViewDriftMonitor, DriftScores
from routing import AdaptationRoute, route_from_signature


@dataclass(frozen=True)
class TRIASDecision:
    scores: DriftScores
    route: AdaptationRoute

    @property
    def components(self):
        return self.route.components


class TRIASController:
    """Controller for Tri-view Robust Incremental Adaptation for Streams (TRIAS)."""
    def __init__(self, monitor: MultiViewDriftMonitor):
        self.monitor = monitor

    def decide(self, x_ref, x_cur, p_ref, p_cur) -> TRIASDecision:
        scores = self.monitor.score(x_ref, x_cur, p_ref, p_cur)
        route = route_from_signature(scores, self.monitor)
        return TRIASDecision(scores=scores, route=route)
