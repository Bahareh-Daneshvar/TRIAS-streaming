import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from counterfactual_routing_diagnostic import ACTIONS, summarize_counterfactuals


def test_counterfactual_summary_and_shuffle_cost_match():
    rows = []
    for i, routed in enumerate(["heads", "medium", "heads", "deep"]):
        row = {"routed_action": routed, "severity_action": "medium"}
        for action in ACTIONS:
            row[f"benefit_{action}"] = 1.0 if action == routed else 0.0
        rows.append(row)
    detail, summary = summarize_counterfactuals(pd.DataFrame(rows), seed=42)
    assert set(summary.policy) == {"routing", "severity_only", "route_shuffle"}
    routing = summary.set_index("policy").loc["routing"]
    assert routing.action_accuracy == 1.0
    assert routing.mean_regret == 0.0
    assert sorted(detail.routed_action) == sorted(detail.shuffled_action)


def test_action_space_covers_frozen_routes():
    assert set(ACTIONS) == {"none", "heads", "gate_heads", "medium", "deep"}
