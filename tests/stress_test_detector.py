import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
from drift import MultiViewDriftMonitor
from synthetic_stream import make_stream
from stream_metrics import detection_delay, false_alarms_before, alarm_rate


def run_one(kind):
    st = make_stream(kind=kind, n_chunks=90, chunk_size=96, dim=12, seed=7)
    # Use first 20 chunks as known pre-shift calibration/reference material.
    x_pool = np.concatenate(st.representations[:20], axis=0)
    p_pool = np.concatenate(st.probabilities[:20], axis=0)
    x_ref = x_pool[-768:]
    p_ref = p_pool[-768:]

    mon = MultiViewDriftMonitor(n_proj=24, seed=11)
    # q=.99 is used in this compact test so 600 draws still give >=6 tail points.
    mon.calibrate_from_pool(
        x_pool, p_pool, window_size=96, n_draws=600,
        mild_q=.90, moderate_q=.97, severe_q=.99, seed=13
    )
    severities, scores = [], []
    for x, p in zip(st.representations[20:], st.probabilities[20:]):
        s = mon.score(x_ref, x, p_ref, p)
        severities.append(s.severity)
        scores.append(s.composite)

    # Onsets relative to the evaluated suffix.
    rel_onsets = tuple(o - 20 for o in st.known_shift_chunks if o >= 20)
    out = {
        "kind": kind,
        "thresholds": mon.thresholds,
        "alarm_rate": alarm_rate(severities),
        "max_score": float(np.max(scores)),
        "severe_count": int(sum(s == "severe" for s in severities)),
    }
    if rel_onsets:
        first = rel_onsets[0]
        out["false_alarms_pre_first_shift"] = false_alarms_before(severities, first)
        out["detection_delay_first_shift"] = detection_delay(severities, first)
    return out


def main():
    results = [run_one(k) for k in ["no_shift", "abrupt", "gradual", "recurring"]]
    # Guardrails are intentionally broad: this is a code stress test, not evidence.
    ns = results[0]
    assert ns["alarm_rate"] < 0.35
    for r in results[1:]:
        assert r["detection_delay_first_shift"] is not None
        assert r["detection_delay_first_shift"] <= 8
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
