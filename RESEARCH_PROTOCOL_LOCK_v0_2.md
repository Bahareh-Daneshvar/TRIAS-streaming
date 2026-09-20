# TRIAS Research Protocol Lock v0.2

## Working research question
Can a financial-text model maintain predictive quality under non-stationary streams more efficiently and reliably when adaptation is triggered by calibrated tri-view, label-free shift evidence and scaled according to shift severity?

## New method
**TRIAS — Tri-view Risk-Informed Adaptive Streaming**

TRIAS is model-agnostic at the controller level. The first study evaluates it on a transparent fusion host, while the journal contribution is the stream monitoring and adaptation mechanism rather than the host backbone.

## Backbone used in the first study: fusion host
- financial-domain contextual text representation
- Bi-GRU contextual branch
- TextCNN local-pattern branch
- dynamic gated fusion
- task output head(s)

The backbone is **not** claimed as new.

## Core contribution hypotheses
### H1 — Tri-view label-free shift monitoring
Use three complementary signals:
1. **semantic representation shift** — sliced Wasserstein distance on latent representations,
2. **predictive-distribution shift** — Jensen–Shannon divergence on probabilistic outputs,
3. **latent dependency shift** — change in correlation/dependency structure.

Each raw signal is calibrated on reference/stationary windows using robust location and scale estimates. A weighted composite score provides the shift severity signal.

### H2 — Severity-aware selective adaptation
- stable: monitor only; no update
- mild: update gate + task head(s)
- moderate: update gate + heads + Bi-GRU/TextCNN
- severe: above + partial encoder update

The method must be evaluated against equal-update and periodic-update controls so that any benefit can be attributed to the severity policy rather than merely to retraining.

### H3 — Streaming-valid evaluation
Primary protocol: chronological **prequential test-then-train** evaluation.

A random split may be used only as a comparability/reconstruction experiment, never as primary evidence for non-stationary streaming claims.

## Calibration thresholds
TRIAS uses three calibrated composite-score thresholds:
- `T_mild`: 90th percentile of stationary calibration composite scores
- `T_moderate`: 97th percentile
- `T_severe`: 99.5th percentile

These are starting values, not fixed scientific truths. Sensitivity analysis must vary them.

Severity mapping:
- `score < T_mild` → stable
- `T_mild <= score < T_moderate` → mild
- `T_moderate <= score < T_severe` → moderate
- `score >= T_severe` → severe

## Minimum baselines
B0 Static fusion host
B1 Fixed-period update
B2 Sliding-window update
B3 Label/error-triggered adaptation (oracle-like delayed-label comparator where labels are available)
B4 Representation-only shift trigger
B5 Equal-update tri-view trigger (no severity selectivity)
B6 **TRIAS**

## Required ablations
A1 remove predictive-distribution view
A2 remove dependency view
A3 remove representation view
A4 fixed equal update after every trigger
A5 no gate updating
A6 no encoder updating under severe shift
A7 unweighted vs calibrated/weighted composite score

## Primary metrics
### Predictive
- macro-F1 (primary)
- balanced accuracy
- per-class recall
- AUROC/AUPRC where scientifically appropriate

### Shift / stream
- detection delay on controlled shift experiments
- false alarms per 1,000 samples
- missed-drift rate where known
- recovery time after a detected shift

### Efficiency
- update frequency
- number/fraction of trainable parameters per update
- adaptation wall-clock time
- inference latency
- cumulative adaptation cost

### Reliability
- Brier score
- expected calibration error (ECE)
- NLL if probabilistic outputs are used for early warning

## Data strategy
### Dataset 1 — primary feasibility stream
Use the public Chinese financial-news dataset cited by the source study. Preserve its date field and rebuild a chronological stream. Do not trust the repository's train/test split for streaming claims until duplicate and temporal leakage audits are complete.

### Dataset 2 — external validation
Select a second public timestamped financial-text dataset only if it provides either human labels or a defensible target. FNSPID is a candidate for temporal-scale validation but its label provenance must be checked before using it as supervised ground truth.

## Phase-1 go/no-go rule
Proceed to the full journal build only if TRIAS achieves at least one of:
1. credible predictive improvement under shift at comparable cost,
2. comparable predictive quality with materially fewer/heavier updates,
3. faster post-shift recovery without unacceptable false alarms.

If none holds, revise the monitor/adaptation mechanism before adding XAI, concept memory, multimodality, or adaptive chunk sizing.

## Deferred extensions
Only after the core result is positive:
- explanation drift / attribution stability
- recurring-regime concept memory
- adaptive chunk size
- multimodal text + market sequence branch
