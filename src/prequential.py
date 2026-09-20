from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Iterable


@dataclass
class ChunkResult:
    chunk_id: int
    n: int
    metric_before_update: float
    drift_severity: str
    drift_score: float
    adapted_components: tuple


def test_then_train_stream(
    chunks: Iterable,
    predict_metric: Callable,
    drift_score: Callable,
    adapt: Callable,
):
    """Generic prequential control loop.

    Each chunk is first evaluated, then drift is scored, then adaptation occurs.
    This ordering prevents the common leakage error of training on a chunk before
    claiming test performance on the same chunk.
    """
    results = []
    for i, chunk in enumerate(chunks):
        metric = float(predict_metric(chunk))
        severity, score = drift_score(chunk)
        comps = tuple(adapt(chunk, severity))
        results.append(ChunkResult(i, len(chunk), metric, severity, float(score), comps))
    return [asdict(r) for r in results]
