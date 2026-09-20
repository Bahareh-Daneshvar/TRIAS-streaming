# TRIAS status v0.4 — 2026-09-20

## Completed this iteration
- Re-audited novelty against current literature.
- Found that generic multi-metric severity-aware adaptation is already published and therefore cannot be the main novelty.
- Reframed TRIAS around **tri-view shift-signature-routed module adaptation**.
- Added per-view calibrated null thresholds and a signature router.
- Added a required severity-only comparator and route-shuffle negative control to the research protocol.
- Corrected fusion host padding handling in both Bi-GRU and TextCNN branches.
- Added explicit Hugging Face encoder wrapper and explicit transformer-block unfreezing.
- Added delayed-label vs label-free claim boundary.
- Added Data-1 sentiment-only claim boundary.
- Added stronger real-data audit: SHA-256, encoding, normalized duplicates, cross-file overlap, temporal duplicates, yearly/monthly density.
- Added null-window calibration safeguards for extreme quantiles.
- Added conservative reference-update sensitivity control.
- Added synthetic no-shift / abrupt / gradual / recurring implementation stress tests.
- All current code tests pass.

## Synthetic implementation checks (not paper evidence)
- no-shift alarm rate in the current stress fixture: ~1.4%
- abrupt shift: first alarm at onset
- gradual shift: first alarm within 2 chunks
- recurring fixture: first alarm at onset

These numbers only verify that the code path behaves sensibly on deliberately strong synthetic perturbations. They must not appear as empirical claims in a manuscript.

## Real-data blocker
The public GitHub repository is web-visible, but this execution container cannot currently retrieve raw GitHub files. The GitHub plugin is not installed/connected yet. No row-level Data-1 result has been fabricated.

## Next executable phase once Data 1 is accessible
1. exact audit + checksums,
2. deduplicated chronological stream manifest,
3. temporal density-based window/chunk choice,
4. initial fusion host sentiment baseline,
5. B0-B7 prequential comparison,
6. go/no-go decision before XAI or concept memory.

## External validation candidate
FinMarBa is currently attractive because the public Hugging Face view exposes dated headlines and market-reaction-derived sentiment labels. It is English, so external validation must use an English financial encoder and should be presented as controller transfer rather than identical-backbone replication.
