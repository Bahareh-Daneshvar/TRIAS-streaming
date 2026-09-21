# TRIAS feasibility pilot decision — v0.8

## Status

**HOLD / NO-GO for the current signature-routing superiority claim.**

This is an engineering feasibility pilot, not a paper result. It uses a compact
reference-only character vocabulary and character embedding host so that the
complete prequential control path can be tested without an external checkpoint.

## Protocol exercised

- exact pinned public Data 1 rows;
- day-grouped chronological reference/calibration/stream partitions;
- 87 complete-day stream chunks covering 11,256 rows;
- frozen sentinel and fixed-reference calibration;
- test-then-train ordering;
- B0-B7 plus the equal-action-multiset route-shuffle control;
- seeds 42, 52, and 62;
- one initial training epoch and one update step per triggered chunk.

## Three-seed summary

| Policy | Macro-F1 mean | Macro-F1 SD | Mean updates | Interpretation |
|---|---:|---:|---:|---|
| B0 static | 0.6969 | 0.0615 | 0.0 | Strong variability across initializations |
| B1 periodic | 0.6921 | 0.0056 | 17.0 | Stable but no aggregate gain |
| B2 sliding | 0.6855 | 0.0239 | 87.0 | Highest update burden; poorer F1 |
| B3 error-triggered | **0.7126** | 0.0151 | 36.0 | Best pilot mean; uses delayed labels |
| B4 representation-only | 0.6741 | 0.0853 | 8.7 | Unstable |
| B5 tri-view equal update | 0.6931 | 0.0488 | 28.0 | No clear gain |
| B6 severity-only | 0.6936 | 0.0388 | 28.0 | Scalar control baseline |
| B7 TRIAS routing | 0.6931 | 0.0387 | 28.0 | No improvement over B6 |
| A9 route-shuffle | 0.6903 | 0.0535 | 28.0 | Same routed-action multiset as B7 |

Paired B7 minus B6 Macro-F1 differences were `+0.000327`, `-0.000697`,
and `-0.001222`. B7 therefore did not provide a meaningful predictive gain.
B7 also lost to the route-shuffle control in two of three seeds.

## Scientific interpretation

The implementation path is executable and the comparison is fair at the
policy level, but this pilot does **not** support claiming that the current
shift signature selects better modules. The result is especially important
because route-shuffle preserves the B7 action multiset: merely changing which
chunk receives which routed action can match or exceed the current routing.

## Boundaries of this conclusion

- Only three seeds were used.
- The compact character host is a feasibility surrogate, not the final encoder.
- One update step may under-express differences between shallow and deep routes.
- The public stream has no independently annotated drift ground truth.
- No inferential significance claim is made.

## Next bounded decision

Do not add XAI, concept memory, or another mechanism. First run a routing
diagnostic that asks whether the three view signatures predict *counterfactual
one-step benefit* for the available component actions. If they do not, retire
the routing-superiority claim and reframe TRIAS around efficient label-free
monitoring with conservative adaptation. If they do, revise only the frozen
routing map and then preregister a new held-out evaluation.
