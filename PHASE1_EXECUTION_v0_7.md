# TRIAS Phase-1 execution record — v0.7

## Execution outcome

The pinned public Data 1 artifact was produced by GitHub Actions run
[`35535231428`](https://github.com/Bahareh-Daneshvar/TRIAS-streaming/actions/runs/35535231428)
from repository commit `82c23d8342714a5b7583a697816f4556c4996b20`.
The workflow verified the audited Git blob identities before publishing the
artifact; raw third-party CSV files remain excluded from the repository.

`src/phase1_prepare.py` then completed successfully on the verified artifact.

| Check | Observed result |
|---|---:|
| Rows | 16,136 |
| Date range | 2016-03-22 to 2019-02-14 |
| Unique calendar days | 246 |
| Reference rows | 3,233 |
| Calibration rows | 1,647 |
| Stream rows | 11,256 |
| Reference end day | 2017-09-12 |
| Calibration end day | 2017-11-23 |
| Days crossing partition boundaries | **0** |

## Class balance

| Partition | Label 0 | Label 1 |
|---|---:|---:|
| Reference | 1,642 | 1,591 |
| Calibration | 785 | 862 |
| Stream | 5,641 | 5,615 |

## Source-file composition

The historical repository split is retained only as provenance. Both files are
combined before the primary chronological partitioning.

| Partition | `train_data.csv` | `test_data.csv` |
|---|---:|---:|
| Reference | 2,758 | 475 |
| Calibration | 1,435 | 212 |
| Stream | 9,543 | 1,713 |

## Validity conclusion

The frozen day-grouped chronological protocol is executable on the exact audited
public data, and the principal boundary-leakage check passes. Phase 1 is closed.
The next scientific gate is fusion host initialization followed by the locked B0-B7
prequential pilot. No new method component should be introduced before that
comparison is available.
