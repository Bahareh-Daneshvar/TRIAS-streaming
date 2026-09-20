# TRIAS status v0.6

## Completed
- GitHub connection verified.
- Exact public Data 1 blobs located and remotely audited.
- Real row-level schema audited through GitHub connector.
- Public-data mismatch with the source manuscript/README documented.
- Historical train/test overlap quantified.
- Day-grouped chronological split frozen for the audited public version.
- Complete-day target-128 chunk policy selected; 64/256 reserved for sensitivity.
- Actual `正负面` header variant added to the loader.
- Git-blob provenance verifier added.
- Day-preserving target chunk iterator added.

## Next executable gate
The dedicated repository and data-preparation workflow are operational. The
pinned CSV artifact was generated, its blob identities were verified, and the
real Phase-1 preparation run completed with zero calendar-day boundary
violations. See `PHASE1_EXECUTION_v0_7.md`.

The next gate is fusion host initialization followed by the locked B0-B7
prequential pilot. No additional method design should be added before that
pilot is run.

## Additional validity fix completed
A potential reject-level confound was removed: the detector no longer measures features from the model it is adapting. A frozen sentinel defines the monitoring space, and fixed-reference threshold calibration now mirrors the operational statistic. Tests cover sentinel independence, causal trailing windows, and fixed-reference calibration.

## GitHub handoff prepared
`.github/workflows/data1-artifact.yml` is ready. On its first push to a dedicated TRIAS repository it downloads the **pinned audited** public CSVs, verifies their Git blob identities, and uploads a GitHub Actions artifact. This avoids committing third-party data into the project repository while giving the connected GitHub tool a downloadable artifact path.
