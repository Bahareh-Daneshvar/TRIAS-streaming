# Novelty audit v0.4 — important course correction

## Why the v0.2/v0.3 novelty framing was not sufficient
A targeted literature search found prior work that substantially overlaps the generic idea of **severity-aware adaptation**.

Most importantly, Shakhovska and Pukach (2025), *Severity-Aware Drift Adaptation for Cost-Efficient Model Maintenance*, aggregate multiple statistical drift measures (including Wasserstein and Jensen-Shannon) into a severity score and use severity to choose no update, incremental update, or full retraining.

A 2026 preprint also proposes parameter-efficient online adaptation that switches between shallow-head updates and deeper LoRA adaptation according to shift/performance severity.

Therefore the journal contribution must **not** be framed as:
> "We combine multiple distances, compute drift severity, and update more layers when severity is higher."

That would be too close to existing work.

## Revised TRIAS novelty hypothesis
TRIAS now focuses on **shift-signature-routed adaptation**, not severity alone.

The detector produces a three-dimensional standardized signature:

`[representation shift, output-distribution shift, dependency-structure shift]`

Severity determines the maximum adaptation budget, while the *shape of the signature* determines where adaptation is applied.

Examples:
- output-dominant shift -> update task head / gate; do not automatically rewrite the language encoder;
- dependency-dominant shift -> update fusion and contextual/local branches;
- representation-dominant severe shift -> permit top-encoder adaptation;
- mixed strong shift -> broader update.

This routing is calibrated against per-view null thresholds learned only from the pre-stream calibration period.

## Why this is more defensible
Existing severity-only strategies answer **how much** adaptation to perform.
TRIAS is intended to answer both:
1. **how much** evidence of shift exists, and
2. **where in the model** adaptation is justified by the observed shift signature.

The scientific claim still requires empirical evidence. The routing table is a hypothesis to test, not a result.

## Required new baseline
Add a severity-only comparator using the **same tri-view detector and same trigger**, but mapping severity directly to fixed update depth.

This is crucial. If TRIAS does not beat this comparator on predictive recovery and/or update cost, the signature-routing contribution is not supported.

## Related-work boundary
Type-aware drift adaptation already exists in general data-stream learning, including methods that identify abrupt/gradual/incremental drift and adapt accordingly. TRIAS must not claim to invent type-aware adaptation broadly.

The narrower claim to test is:
> using **where-shift-appears across semantic representation, model output, and latent dependency views** to route adaptation across modules of a streaming financial-text model.

That is the contribution target for the present study.
