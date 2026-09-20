from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from drift import DriftScores, MultiViewDriftMonitor


@dataclass(frozen=True)
class AdaptationRoute:
    severity: str
    dominant_view: str
    components: tuple[str, ...]
    rationale_code: str


def _relative_signature(scores: DriftScores, monitor: MultiViewDriftMonitor):
    if monitor.z_high is None:
        raise RuntimeError("Monitor must be calibrated before routing")
    z = np.array([scores.z_representation, scores.z_output, scores.z_dependency], dtype=float)
    denom = np.maximum(np.asarray(monitor.z_high, dtype=float), 1e-9)
    return z / denom


def route_from_signature(scores: DriftScores, monitor: MultiViewDriftMonitor, dominance_ratio: float = 1.25):
    """Route updates using both magnitude and *where* the shift appears.

    Severity defines the maximum adaptation budget. The normalized tri-view
    signature decides which modules receive that budget. This is intentionally
    different from a scalar severity-only rule.
    """
    if scores.severity == "stable":
        return AdaptationRoute("stable", "none", tuple(), "stable_no_update")

    rel = _relative_signature(scores, monitor)
    names = ["representation", "output", "dependency"]
    order = np.argsort(rel)[::-1]
    top, second = rel[order[0]], rel[order[1]]
    dominant = names[int(order[0])] if top >= dominance_ratio * max(second, 1e-9) else "mixed"

    # Mild: cheapest plausible action; never touch feature extractor/encoder.
    if scores.severity == "mild":
        if dominant == "output":
            return AdaptationRoute("mild", dominant, ("heads",), "mild_output_head_only")
        return AdaptationRoute("mild", dominant, ("gate", "heads"), "mild_fusion_or_mixed")

    # Moderate: adapt local/context branches only when representation/dependency
    # evidence dominates; output-only shifts remain cheap.
    if scores.severity == "moderate":
        if dominant == "output":
            return AdaptationRoute("moderate", dominant, ("gate", "heads"), "moderate_output_no_deep_update")
        return AdaptationRoute(
            "moderate", dominant,
            ("gate", "heads", "bigru", "textcnn"),
            "moderate_representation_dependency_or_mixed",
        )

    # Severe: permit encoder adaptation only when representation shift itself is
    # above its calibrated 99th-percentile null threshold. A severe output-prior
    # change alone should not automatically rewrite the language encoder.
    rep_high = scores.z_representation >= float(monitor.z_high[0])
    if dominant == "output" and not rep_high:
        return AdaptationRoute("severe", dominant, ("gate", "heads"), "severe_output_without_rep_shift")
    if dominant == "dependency" and not rep_high:
        return AdaptationRoute(
            "severe", dominant,
            ("gate", "heads", "bigru", "textcnn"),
            "severe_dependency_without_rep_shift",
        )
    components = ["gate", "heads", "bigru", "textcnn"]
    if rep_high:
        components.append("encoder_top")
        code = "severe_with_representation_shift"
    else:
        code = "severe_mixed_no_representation_threshold"
    return AdaptationRoute("severe", dominant, tuple(components), code)
