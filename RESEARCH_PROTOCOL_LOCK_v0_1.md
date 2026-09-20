# Research Protocol Lock v0.1

## Working research question
Can a financial-text risk model maintain predictive quality under non-stationary streams more efficiently and transparently when adaptation is triggered by multi-view, label-free drift evidence and scaled according to drift severity?

## Backbone
Transparent fusion-host implementation based on the supplied architecture description:
- financial RoBERTa representation
- Bi-GRU contextual branch
- TextCNN local-pattern branch
- dynamic gated fusion
- multi-task outputs

The backbone is **not** itself claimed as new.

## Core novelty to test
### C1. Multi-view label-free drift score
Three views:
1. semantic representation drift (sliced Wasserstein)
2. predictive distribution drift (Jensen-Shannon divergence)
3. latent dependency drift (correlation-structure change)

The views are robustly calibrated on a stationary/reference segment and combined into a composite severity score.

### C2. Severity-aware selective adaptation
- stable: no model update
- mild: gate + heads
- moderate: gate + heads + Bi-GRU/TextCNN
- severe: above + partial encoder update

This is the main mechanism that must beat both static and unconditional/periodic retraining baselines on the accuracy-compute tradeoff.

### C3. Streaming-valid evaluation
Primary protocol: chronological prequential test-then-train.
Random split is retained only to reproduce/compare against the original paper, never as the main evidence for streaming claims.

## Minimum baselines
B0 Static fusion host
B1 Periodic retraining at fixed intervals
B2 Sliding-window retraining
B3 Error-triggered detector + adaptation (when labels are available)
B4 Representation-only drift trigger
B5 Multi-view fixed-policy fusion host

## Required ablations
A1 remove output drift view
A2 remove dependency drift view
A3 remove representation drift view
A4 equal/fixed update regardless of severity
A5 no gate updating
A6 no encoder updating under severe drift

## Primary metrics
Predictive:
- macro-F1 (primary because class imbalance is expected)
- balanced accuracy
- per-class recall

Streaming/drift:
- detection delay on controlled drift experiments
- false alarm rate / alarms per 1,000 samples
- missed-drift rate where ground truth is known

Efficiency:
- update frequency
- trainable parameter count per update regime
- wall-clock adaptation time
- inference latency

Reliability:
- Brier score / ECE when probabilities are used for early warning

## Data strategy
### Dataset 1
Use the public Chinese financial-news dataset named in the paper, preserving its date field and reconstructing a chronological stream.

### Dataset 2
Do not depend on the paper's private/self-collected Data 2. Select a second public timestamped financial-text dataset for external validation.

## Stop/go criteria after Phase 1 experiments
Proceed to a full journal manuscript only if the proposed method shows at least one of:
1. statistically credible macro-F1 improvement under drift at similar compute, OR
2. similar predictive performance with materially fewer/heavier updates, OR
3. materially lower recovery delay after drift without unacceptable false alarms.

If none is achieved, revise the detector/adaptation mechanism before adding XAI or concept memory.

## Deferred extensions
Only after the core result is positive:
- explanation drift / attribution stability
- concept memory for recurring regimes
- adaptive chunk size
- multimodal market-price branch
