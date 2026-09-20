# TRIAS Research Protocol Lock v0.3

## Working research question
Can a financial-text classifier maintain predictive quality under non-stationary streams more efficiently and reliably when adaptation is triggered by calibrated tri-view, label-free distribution-shift evidence and scaled according to shift severity?

## Method
**TRIAS — Tri-view Robust Incremental Adaptation for Streams**

TRIAS is model-agnostic at the controller level. The first study evaluates it on a transparent fusion host, while the journal contribution is the stream monitoring and adaptation mechanism rather than the host backbone.

## Terminology lock: shift is not automatically concept drift
The label-free detector observes changes in representations, model-output distributions, and latent dependencies. These are evidence of **distribution shift / non-stationarity**. They do not by themselves prove strict concept drift, i.e. a change in P(Y|X).

Use the term **concept drift** only when:
- labels permit evidence of changing predictive relationships, or
- a controlled synthetic experiment defines the drift ground truth.

This wording prevents a common overclaim in drift papers.

## Data-label contract
### Public Data 1
The public Chinese financial-news dataset has sentiment labels. It does not supply the source paper's self-collected risk-level and risk-type labels.

Therefore, the primary Data-1 experiment uses the **sentiment head only**. Risk-level/type heads remain architecture-compatible but are not evaluated or claimed on Data 1.

### Online adaptation contract
The **trigger is label-free**.

The primary benchmark uses **delayed-label test-then-train adaptation**:
1. predict current chunk,
2. record metrics,
3. score label-free shift,
4. reveal labels for benchmark adaptation,
5. selectively update permitted components.

This is not described as fully label-free learning. A fully unlabeled update rule (self-supervised/pseudo-label) is deferred unless the core delayed-label study is positive.

## Core contribution hypotheses
### H1 — Tri-view label-free shift monitoring
Three complementary signals:
1. semantic representation shift — sliced Wasserstein distance,
2. predictive-distribution shift — Jensen-Shannon divergence on probabilistic outputs,
3. latent dependency shift — change in correlation/dependency structure.

Each is robustly standardized using median/MAD from a pre-stream calibration interval.

**Default composite weights are equal after standardization (1/3, 1/3, 1/3).** Alternative weights are sensitivity/ablation settings. No weights may be tuned on the held-out stream.

### H2 — Severity-aware selective adaptation
- stable: no update
- mild: gate + task head(s)
- moderate: gate + heads + Bi-GRU/TextCNN
- severe: above + partial encoder update

The exact layer mapping is fixed before the primary test run.

### H3 — Streaming-valid evaluation
Primary protocol: chronological **prequential test-then-train** evaluation.

Random split results, if reproduced, are comparability-only and cannot support streaming/non-stationarity claims.

## Threshold calibration
Thresholds are estimated from the pre-stream calibration period only.

Starting quantiles:
- T_mild = Q0.90
- T_moderate = Q0.97
- T_severe = Q0.995

Because extreme empirical quantiles are unstable with few windows, TRIAS v0.3 uses contiguous-window null resampling within the calibration interval to create >=2,000 calibration comparisons for Q0.995. Resampling draws may overlap across draws and are **not** treated as independent observations; they are only an empirical threshold-estimation device.

Sensitivity analysis must vary quantiles and window size.

## Reference policy
Primary comparison uses a fixed pre-stream reference for clean attribution of detector behavior.

A conservative adaptive-reference sensitivity analysis is allowed: reference updates only after several consecutive stable chunks and never immediately after a detected shift. Reference maintenance is an implementation control, not a novelty claim.

## Minimum baselines
B0 Static fusion host
B1 Fixed-period update
B2 Sliding-window update
B3 Label/error-triggered adaptation (delayed-label comparator)
B4 Representation-only shift trigger
B5 Equal-update tri-view trigger (same trigger, no severity selectivity)
B6 TRIAS

## Required ablations
A1 remove predictive-output view
A2 remove dependency view
A3 remove representation view
A4 fixed equal update after every trigger
A5 no gate updating
A6 no encoder updating under severe shift
A7 equal vs alternative pre-specified composite weights
A8 fixed vs conservative adaptive reference

## Primary metrics
Predictive: macro-F1, balanced accuracy, per-class recall, AUROC/AUPRC where appropriate.

Shift/stream: detection delay on controlled shifts, false alarms per 1,000 samples, missed-shift rate, recovery time.

Efficiency: update frequency, trainable-parameter fraction, adaptation wall-clock time, inference latency, cumulative adaptation cost.

Reliability: Brier score, ECE, NLL.

## Data strategy
### Dataset 1 — primary feasibility stream
Use the public Chinese financial-news dataset cited by the source paper. Preserve its date field and rebuild chronological order. The repository train/test split is not trusted for streaming claims until duplicate/date leakage audit is complete.

### Dataset 2 — external validation
A second timestamped dataset must have defensible targets. Current candidates:
- FinMarBa for market-reaction-based sentiment with timestamps,
- FNSPID mainly for large-scale temporal/unsupervised shift validation unless a defensible supervised target is established.

Language/backbone mismatch must be handled explicitly; do not pretend a Chinese RoBERTa backbone transfers unchanged to English data.

## Phase-1 go/no-go rule
Proceed to XAI/concept-memory extensions only if TRIAS achieves at least one of:
1. credible predictive improvement under shift at comparable cost,
2. comparable predictive quality with materially fewer/heavier updates,
3. faster post-shift recovery without unacceptable false alarms.

If none holds, revise the core monitor/adaptation policy before adding complexity.
