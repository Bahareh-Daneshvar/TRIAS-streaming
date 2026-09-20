# Method Name Lock v0.2

## Working method name
**TRIAS** — **Tri-view Risk-Informed Adaptive Streaming**

### Full manuscript-style description
**TRIAS: a tri-view, severity-aware adaptive streaming framework for non-stationary financial text risk modelling.**

## Why this name
TRIAS names the *new method*, not the host neural backbone. It therefore remains independent of any source-model or prior-draft terminology.

The three monitoring views are:
1. semantic representation shift,
2. predictive-distribution shift,
3. latent dependency shift.

Their calibrated evidence controls a severity-aware adaptation policy.

## Naming boundary
- **TRIAS** = the new stream-monitoring + severity-aware adaptation method.
- **fusion host** = the transparent host classifier used to evaluate the controller.
- The paper must not imply that Bi-GRU, TextCNN, attention, dynamic gating, or the fusion backbone were invented by TRIAS.

## Current novelty claim (provisional until full literature review)
TRIAS is being tested as a combination of multi-view label-free shift evidence and severity-conditioned selective model adaptation under chronological/prequential evaluation.

Do **not** claim “first” or “novel” for the individual distance measures, the idea of concept-drift adaptation, or severity estimation until the final literature audit is complete.

## Name collision note
A quick web literature check on 20 September 2026 found no directly relevant concept-drift / financial-NLP method named “TRIAS”. This is not a trademark search and should be rechecked immediately before submission.
