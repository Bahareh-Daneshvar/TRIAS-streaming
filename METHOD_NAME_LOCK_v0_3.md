# Method name lock v0.3

## Working name
**TRIAS — Tri-view Robust Incremental Adaptation for Streams**

This expansion replaces the earlier working expansion "Tri-view Risk-Informed Adaptive Streaming".

## Why this is better
1. It is independent of inherited source-model terminology.
2. It describes the methodological contribution rather than the application domain.
3. It does not imply that risk-level/risk-type labels exist in every dataset. The public Data 1 used by the source paper provides sentiment labels, whereas the source paper's additional risk labels are tied to its non-public self-collected Data 2.
4. It keeps TRIAS model-agnostic: the first host is a fusion classifier, but the controller can later be tested with another encoder/classifier.

## Claim boundary
TRIAS refers to the controller consisting of:
- tri-view shift monitoring,
- calibrated severity assignment,
- selective incremental adaptation policy,
- streaming-valid control logic.

It does **not** refer to the encoder, Bi-GRU, TextCNN, attention, or fusion gate. Those host components are not claimed as new.

## Naming caution
A web collision check found uses of the word/name "TRIAS" in unrelated domains and as a person's given name, but no exact match for the expansion above in concept-drift / streaming-ML literature. This is a working research name, not a trademark clearance.
