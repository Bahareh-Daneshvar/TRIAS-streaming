# TRIAS validity lock v0.6

## Problem caught before pilot
If the same adaptive network supplies drift representations before and after an update, changes in its parameters can alter the feature space even when the incoming data distribution is unchanged. A detector can then alarm on its own adaptation.

## Locked solution
- Freeze a pre-stream **sentinel** copy after reference training.
- Compute representation, dependency and output-distribution monitoring views only from this frozen sentinel.
- Keep the adaptive predictor separate.
- Use a 128-row causal trailing monitor window at complete-day chunk boundaries.
- Calibrate thresholds with the exact same fixed-reference statistic on contiguous pre-stream calibration windows.

This avoids adaptation-induced detector contamination and makes B6 versus TRIAS routing comparisons interpretable.
