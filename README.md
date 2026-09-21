# TRIAS Journal Upgrade v0.6

**Current stage:** Phase 1 is complete and the first full-stream, three-seed B0-B7 feasibility pilot is executable. The current routing-superiority claim is on HOLD because B7 did not improve on B6; see `PILOT_GO_NO_GO_v0_8.md`.

See `DATA1_REMOTE_AUDIT_v0_6.md`, `PHASE1_EXECUTION_v0_7.md`, and `STATUS_v0_6.md`.

**TRIAS = Tri-view Robust Incremental Adaptation for Streams**

TRIAS is a model-agnostic controller tested first on a transparent **fusion host** classifier. Its current novelty target is **tri-view shift-signature-routed module adaptation**, not generic severity-aware retraining.

## Why v0.4 changed direction
A 2025 paper already combines multiple statistical drift measures into a severity score and uses severity to choose incremental vs full retraining. A 2026 preprint also uses severity/performance signals for head-vs-deeper parameter-efficient adaptation. Therefore severity-aware updating alone is not a defensible novelty claim.

TRIAS v0.4 instead uses standardized evidence from:
1. semantic representation shift,
2. predictive-output distribution shift,
3. latent dependency shift,

to decide **where** adaptation should occur. Severity constrains the budget; the shift signature routes that budget.

## Scientific guardrails
- distribution shift != automatically strict concept drift;
- trigger/routing are label-free, primary adaptation uses delayed labels after prediction;
- Data 1 is sentiment-only;
- the exact source encoder checkpoint is not identifiable, so the implementation is described only as a transparent fusion host;
- streaming evidence is chronological prequential, not random CV;
- severity-only adaptation is a required baseline;
- route-shuffle at equal update cost is a required negative control.

## Code
- `src/fusion_backbone.py`: padding-safe fusion host branches/gating
- `src/hf_fusion.py`: optional Hugging Face encoder wrapper + explicit block adaptation
- `src/drift.py`: tri-view shift monitor + null calibration
- `src/routing.py`: signature router
- `src/reference.py`: conservative reference sensitivity control
- `src/data1.py`, `src/data_audit.py`: public Data-1 normalization/audit
- `src/train_utils.py`: training/prediction utilities
- `src/synthetic_stream.py`, `src/stream_metrics.py`: controlled implementation stress tests
- `src/char_fusion.py`, `src/pilot_b0_b7.py`: explicitly non-final B0-B7 feasibility pilot

## Checks
```bash
python tests/smoke_test.py
python tests/stress_test_detector.py
python tests/test_reference_manager.py
python tests/test_padding_invariance.py
python tests/test_signature_routing.py
```

Synthetic checks are implementation tests, not paper results.

## v0.5 phase-1 readiness
- Added `src/preflight.py` to make the only external blocker explicit.
- Added `src/phase1_prepare.py` for end-to-end audit + chronology preparation.
- Added day-grouped temporal boundaries so one calendar day can never leak across reference/calibration/stream partitions.
- The historical repository train/test split is retained only as source metadata for contamination/reproduction checks; it is not the primary streaming split.
- Added pytest-discoverable core tests and temporal-boundary tests.
