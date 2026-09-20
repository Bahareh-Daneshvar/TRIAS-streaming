# TRIAS Research Protocol Lock v0.4

## Working name
**TRIAS — Tri-view Robust Incremental Adaptation for Streams**

## Research question
Can a streaming financial-text classifier recover from non-stationarity more efficiently when a calibrated tri-view shift signature is used not only to trigger adaptation, but also to route updates to the model components most plausibly implicated by the observed shift?

## Terminology
The label-free detector provides evidence of **distribution shift / non-stationarity**. Do not call every alert strict concept drift. Strict concept drift P(Y|X) requires label evidence or controlled ground truth.

## Host model
The fusion host is a transparent evaluation backbone:
- contextual encoder,
- Bi-GRU + attention branch,
- TextCNN local branch,
- dynamic gate,
- task head.

The host architecture is inherited and not claimed as new.

## Data-1 task boundary
Public Data 1 supports sentiment classification only. Risk-level/type heads are not evaluated without valid corresponding labels.

## Adaptation supervision boundary
TRIAS **detection/routing is label-free**.
Primary adaptation benchmark is delayed-label prequential:
1. predict,
2. record performance,
3. compute label-free shift signature,
4. reveal benchmark labels,
5. adapt only the routed components.

Do not describe the whole learner as fully label-free.

## Tri-view shift signature
For chunk t, compute robust-standardized evidence:
- z_rep: sliced-Wasserstein representation shift,
- z_out: Jensen-Shannon output-distribution shift,
- z_dep: latent-dependency shift.

Default composite weights are equal after standardization. Alternative weights are pre-specified sensitivity settings only.

## Calibrated thresholds
Composite severity thresholds start at Q0.90 / Q0.97 / Q0.995 from the pre-stream calibration interval.
Per-view Q0.99 null thresholds are also stored for routing.
Extreme quantiles require enough null comparisons; v0.4 uses contiguous-window null resampling with >=2,000 draws for the final Q0.995 configuration.

## Novelty target: signature-routed adaptation
**Severity-only adaptation is a baseline, not the main contribution.**

Severity defines the maximum update budget. Relative per-view evidence routes the budget:
- output-dominant: favor head/gate adaptation;
- dependency-dominant: favor gate + Bi-GRU/TextCNN adaptation;
- representation-dominant: permit deeper feature/encoder adaptation only when representation evidence exceeds its calibrated high-null threshold;
- mixed: broader adaptation within the severity budget.

This routing table is frozen before primary test evaluation and must be challenged by ablations.

## Primary baselines
B0 Static fusion host
B1 Fixed-period update
B2 Sliding-window update
B3 Delayed-label error-triggered update
B4 Representation-only trigger
B5 Tri-view trigger + equal full/medium update
B6 Tri-view trigger + **severity-only** fixed-depth policy
B7 **TRIAS: tri-view signature-routed adaptation**

## Required ablations
A1 remove output view
A2 remove dependency view
A3 remove representation view
A4 severity-only routing
A5 no gate update
A6 no encoder update
A7 fixed vs adaptive reference
A8 equal vs pre-specified alternative view weights
A9 route-shuffle negative control: randomly permute routed actions while preserving update counts

A9 is especially important: it tests whether *which module* is updated matters beyond simply spending the same compute budget.

## Metrics
Predictive: macro-F1 (primary), balanced accuracy, per-class recall.
Reliability: Brier, ECE, NLL.
Streaming: detection delay on controlled shifts, false alarms/1,000, missed shifts, recovery time.
Efficiency: update count, trainable-parameter fraction, adaptation wall time, inference latency, cumulative adaptation cost.

## Reference policy
Primary detector analysis uses fixed pre-stream reference for attribution.
Conservative stable-only reference refresh is a sensitivity analysis, not novelty.

## Data strategy
Data 1: public Chinese dataset from source paper, chronological after row-level audit.
External: FinMarBa is currently the strongest supervised timestamped candidate but is English, so it requires an English financial encoder and demonstrates controller-level transfer, not backbone identity. FNSPID is useful for temporal-scale validation but supervised label provenance must be defined before classification claims.

## Go/no-go
Proceed to XAI/concept memory only if B7 beats or matches B6 with a meaningful efficiency/recovery benefit. If signature routing adds no value over severity-only adaptation, revise or abandon that claim before expanding scope.

## v0.6 validity lock: detector/predictor separation
TRIAS uses a **frozen pre-stream sentinel** for all three monitoring views. The adaptive predictor is not allowed to define the detector's representation space after adaptation begins. This prevents model parameter updates from being misread as data distribution shift.

Operational monitoring uses a fixed-size trailing window of 128 already-seen samples at each complete-day chunk boundary. Adaptation chunks remain complete-day grouped and target 128 rows, so no within-day pseudo-order is manufactured.

Primary threshold calibration mirrors the deployed statistic: the fixed sentinel reference is compared against every contiguous 128-row window in the pre-stream calibration interval (stride 1). With 1,647 calibration rows this yields 1,520 empirical null comparisons, sufficient for the pre-registered 0.995 severe quantile without inventing 2,000 pseudo-independent draws.

The sentinel separation is a validity safeguard, not a standalone novelty claim.
