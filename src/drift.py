from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def _safe_standardize(x, center, scale):
    return (x - center) / np.maximum(scale, 1e-12)


def sliced_wasserstein(x_ref: np.ndarray, x_cur: np.ndarray, n_proj: int = 32, seed: int = 0) -> float:
    """Mean 1D Wasserstein distance over deterministic random projections."""
    x_ref = np.asarray(x_ref, dtype=float)
    x_cur = np.asarray(x_cur, dtype=float)
    if x_ref.ndim != 2 or x_cur.ndim != 2 or x_ref.shape[1] != x_cur.shape[1]:
        raise ValueError("Representation arrays must be [n_samples, n_features] with matching feature dimension")
    if min(len(x_ref), len(x_cur)) < 2:
        raise ValueError("At least two samples are required per window")
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, x_ref.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return float(np.mean([wasserstein_distance(x_ref @ d, x_cur @ d) for d in dirs]))


def output_js(p_ref: np.ndarray, p_cur: np.ndarray) -> float:
    """Jensen-Shannon divergence between mean predictive class distributions.

    This measures model-output distribution shift. It does *not* establish real
    concept drift P(Y|X) without labels.
    """
    pr = np.asarray(p_ref, dtype=float)
    pc = np.asarray(p_cur, dtype=float)
    if pr.ndim != 2 or pc.ndim != 2 or pr.shape[1] != pc.shape[1]:
        raise ValueError("Probability arrays must be [n_samples, n_classes] with matching class dimension")
    pr = np.clip(pr.mean(axis=0), 1e-12, None)
    pc = np.clip(pc.mean(axis=0), 1e-12, None)
    pr /= pr.sum()
    pc /= pc.sum()
    return float(jensenshannon(pr, pc, base=2.0) ** 2)


def dependency_drift(x_ref: np.ndarray, x_cur: np.ndarray) -> float:
    """Normalized Frobenius distance between latent correlation structures."""
    def corr(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError("Representations must be 2D")
        if x.shape[0] < 3:
            return np.eye(x.shape[1])
        c = np.corrcoef(x, rowvar=False)
        return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

    a, b = corr(x_ref), corr(x_cur)
    if a.shape != b.shape:
        raise ValueError("Reference and current dependency matrices must match")
    return float(np.linalg.norm(a - b, ord="fro") / max(1, a.shape[0]))


@dataclass(frozen=True)
class DriftScores:
    representation: float
    output: float
    dependency: float
    z_representation: float
    z_output: float
    z_dependency: float
    composite: float
    severity: str


class MultiViewDriftMonitor:
    """Tri-view label-free distribution-shift monitor used by TRIAS.

    The three views are standardized using robust median/MAD estimates from a
    *pre-stream calibration period*. Default weights are equal after
    standardization; alternative weights belong in sensitivity analysis, not in
    post-hoc test-set tuning.
    """

    def __init__(self, weights=(1 / 3, 1 / 3, 1 / 3), n_proj=32, seed=0):
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (3,):
            raise ValueError("weights must contain exactly three values")
        if np.any(self.weights < 0) or self.weights.sum() <= 0:
            raise ValueError("weights must be non-negative and sum to > 0")
        self.weights /= self.weights.sum()
        self.n_proj = int(n_proj)
        self.seed = int(seed)
        self.center = None
        self.scale = None
        self.t_mild = None
        self.t_moderate = None
        self.t_severe = None
        self.z_high = None
        self.calibration_meta = None

    def raw(self, x_ref, x_cur, p_ref, p_cur):
        return np.array([
            sliced_wasserstein(x_ref, x_cur, self.n_proj, self.seed),
            output_js(p_ref, p_cur),
            dependency_drift(x_ref, x_cur),
        ], dtype=float)

    def calibrate(self, raw_scores: np.ndarray, mild_q=0.90, moderate_q=0.97, severe_q=0.995):
        r = np.asarray(raw_scores, dtype=float)
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError("raw_scores must be [n_windows, 3]")
        if not (0 < mild_q < moderate_q < severe_q < 1):
            raise ValueError("Require 0 < mild_q < moderate_q < severe_q < 1")
        if len(r) < 50:
            raise ValueError("Use at least 50 null/calibration comparisons")
        expected_severe_tail = len(r) * (1.0 - severe_q)
        if expected_severe_tail < 5:
            raise ValueError(
                f"Severe quantile is too extreme for {len(r)} calibration comparisons; "
                f"need at least {math.ceil(5/(1-severe_q))} for >=5 expected tail observations"
            )
        self.center = np.median(r, axis=0)
        mad = np.median(np.abs(r - self.center), axis=0)
        self.scale = 1.4826 * mad + 1e-9
        z = np.maximum(0.0, _safe_standardize(r, self.center, self.scale))
        comp = z @ self.weights
        self.t_mild = float(np.quantile(comp, mild_q))
        self.t_moderate = float(np.quantile(comp, moderate_q))
        self.t_severe = float(np.quantile(comp, severe_q))
        # Per-view high-null thresholds used by the signature router.
        self.z_high = np.quantile(z, 0.99, axis=0).astype(float)
        self.calibration_meta = {
            "n": int(len(r)),
            "mild_q": float(mild_q),
            "moderate_q": float(moderate_q),
            "severe_q": float(severe_q),
        }
        return self

    def calibrate_from_pool(
        self,
        x_pool: np.ndarray,
        p_pool: np.ndarray,
        window_size: int,
        n_draws: int = 2000,
        mild_q: float = 0.90,
        moderate_q: float = 0.97,
        severe_q: float = 0.995,
        seed: int | None = None,
    ):
        """Estimate the null score distribution from a pre-stream calibration pool.

        Each draw compares two contiguous windows sampled from the calibration
        interval. Contiguity preserves local temporal structure better than IID
        row resampling. Draws may reuse rows across draws; they are used only for
        threshold estimation, never as independent inferential observations.
        """
        x = np.asarray(x_pool, dtype=float)
        p = np.asarray(p_pool, dtype=float)
        if x.ndim != 2 or p.ndim != 2 or len(x) != len(p):
            raise ValueError("x_pool and p_pool must be aligned 2D arrays")
        w = int(window_size)
        if w < 8:
            raise ValueError("window_size must be >= 8")
        if len(x) < 3 * w:
            raise ValueError("Calibration pool should contain at least 3 x window_size samples")
        if n_draws < 50:
            raise ValueError("n_draws must be >= 50")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        max_start = len(x) - w
        raw_scores = []
        attempts = 0
        max_attempts = n_draws * 50
        while len(raw_scores) < n_draws and attempts < max_attempts:
            attempts += 1
            a = int(rng.integers(0, max_start + 1))
            b = int(rng.integers(0, max_start + 1))
            # Avoid heavily overlapping windows; otherwise null scores collapse.
            if abs(a - b) < max(1, w // 2):
                continue
            raw_scores.append(self.raw(x[a:a+w], x[b:b+w], p[a:a+w], p[b:b+w]))
        if len(raw_scores) < n_draws:
            raise RuntimeError("Could not sample enough weakly-overlapping calibration window pairs")
        self.calibrate(np.vstack(raw_scores), mild_q, moderate_q, severe_q)
        self.calibration_meta.update({
            "mode": "contiguous-window null resampling",
            "window_size": w,
            "pool_size": int(len(x)),
            "draws": int(n_draws),
        })
        return self

    def calibrate_fixed_reference(
        self,
        x_ref: np.ndarray,
        p_ref: np.ndarray,
        x_cal: np.ndarray,
        p_cal: np.ndarray,
        window_size: int = 128,
        stride: int = 1,
        mild_q: float = 0.90,
        moderate_q: float = 0.97,
        severe_q: float = 0.995,
    ):
        """Calibrate the exact operational statistic against a fixed reference.

        Primary TRIAS monitoring compares a frozen pre-stream sentinel reference
        to a fixed-size trailing monitor window. Threshold calibration should
        therefore use the same statistic, rather than calibration-vs-calibration
        window pairs whose variance is different. Overlapping calibration windows
        are allowed because these scores estimate empirical thresholds and are not
        treated as independent inferential observations.
        """
        xr = np.asarray(x_ref, dtype=float)
        pr = np.asarray(p_ref, dtype=float)
        xc = np.asarray(x_cal, dtype=float)
        pc = np.asarray(p_cal, dtype=float)
        if xr.ndim != 2 or pr.ndim != 2 or xc.ndim != 2 or pc.ndim != 2:
            raise ValueError("reference/calibration arrays must be 2D")
        if len(xr) != len(pr) or len(xc) != len(pc):
            raise ValueError("representations and probabilities must align")
        if xr.shape[1] != xc.shape[1] or pr.shape[1] != pc.shape[1]:
            raise ValueError("reference/calibration feature dimensions must match")
        w = int(window_size)
        st = int(stride)
        if w < 8 or st < 1:
            raise ValueError("window_size must be >=8 and stride >=1")
        if len(xc) < w:
            raise ValueError("calibration pool is smaller than the monitor window")
        raw_scores = [
            self.raw(xr, xc[i:i+w], pr, pc[i:i+w])
            for i in range(0, len(xc) - w + 1, st)
        ]
        self.calibrate(np.vstack(raw_scores), mild_q, moderate_q, severe_q)
        self.calibration_meta.update({
            "mode": "fixed-reference contiguous-calibration windows",
            "window_size": w,
            "stride": st,
            "reference_size": int(len(xr)),
            "calibration_pool_size": int(len(xc)),
            "comparisons": int(len(raw_scores)),
        })
        return self

    @property
    def thresholds(self):
        if self.center is None:
            return None
        return {
            "mild": self.t_mild,
            "moderate": self.t_moderate,
            "severe": self.t_severe,
            "view_high_99": {
                "representation": float(self.z_high[0]),
                "output": float(self.z_high[1]),
                "dependency": float(self.z_high[2]),
            },
        }

    def score(self, x_ref, x_cur, p_ref, p_cur) -> DriftScores:
        if self.center is None:
            raise RuntimeError("Calibrate the monitor before scoring")
        r = self.raw(x_ref, x_cur, p_ref, p_cur)
        z = np.maximum(0.0, _safe_standardize(r, self.center, self.scale))
        c = float(z @ self.weights)
        if c >= self.t_severe:
            sev = "severe"
        elif c >= self.t_moderate:
            sev = "moderate"
        elif c >= self.t_mild:
            sev = "mild"
        else:
            sev = "stable"
        return DriftScores(
            float(r[0]), float(r[1]), float(r[2]),
            float(z[0]), float(z[1]), float(z[2]),
            c, sev
        )
