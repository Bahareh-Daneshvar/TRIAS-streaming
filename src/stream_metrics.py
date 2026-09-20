from __future__ import annotations
import numpy as np

SEVERITY_ORDER = {"stable": 0, "mild": 1, "moderate": 2, "severe": 3}


def first_alarm_after(severities, onset, min_severity="mild"):
    threshold = SEVERITY_ORDER[min_severity]
    for i in range(int(onset), len(severities)):
        if SEVERITY_ORDER[severities[i]] >= threshold:
            return i
    return None


def detection_delay(severities, onset, min_severity="mild"):
    first = first_alarm_after(severities, onset, min_severity)
    return None if first is None else int(first - onset)


def false_alarms_before(severities, onset, min_severity="mild"):
    threshold = SEVERITY_ORDER[min_severity]
    return int(sum(SEVERITY_ORDER[s] >= threshold for s in severities[:int(onset)]))


def alarm_rate(severities, min_severity="mild"):
    threshold = SEVERITY_ORDER[min_severity]
    if not severities:
        return 0.0
    return float(np.mean([SEVERITY_ORDER[s] >= threshold for s in severities]))
