# Data 1 remote audit v0.6

## Provenance
The audit was run against the public GitHub repository `wwwxmu/Dataset-of-financial-news-sentiment-classification` using the GitHub connector, not a locally renamed copy.

- audited repository head: `9e3166942fed246c193ec67996d5294debfd71e3`
- `train_data.csv` Git blob: `060c4c28174d6f7c62237af625266ffa76455d9f`
- `test_data.csv` Git blob: `9ddd2c7e5b3fa87859939db430a93cbc41e644f5`

## Critical reproducibility finding
The two accessible CSVs contain **16,136 rows**, exactly balanced between labels 0 and 1 (**8,068 each**). This does **not** match the repository README and the source manuscript, which describe a final set of **17,149** samples with **12,514 positive** and **4,635 negative** examples.

This mismatch is now treated as a reproducibility limitation. TRIAS will not silently claim to reproduce the manuscript's Data 1 sample composition. The primary experiment will use the exact accessible public CSV version identified by the blob hashes above.

## Accessible CSV audit
- train: 13,736 rows; 6,868/6,868 class balance; 0 date parse failures.
- test: 2,400 rows; 1,200/1,200 class balance; 0 date parse failures.
- combined date span: 2016-03-22 to 2019-02-14 across 246 observed dates.
- exact semantic duplicate extras (date/company/code/label/title/body): 268 combined.
- normalized title+body duplicate extras: 275 combined.
- train/test overlap: 671 unique normalized titles and 101 unique normalized title+body fingerprints occur in both files.

The historical train/test split is therefore unsuitable as the primary streaming protocol. The files are combined, source identity is retained for contamination analysis, and chronology is reconstructed from dates.

## Locked day-grouped chronological partition
Using row targets of 20% reference and 10% calibration while forbidding a calendar date from crossing a boundary gives:

- reference: through **2017-09-12**, 3,233 rows (1,642 negative; 1,591 positive)
- calibration: **2017-09-13 to 2017-11-23**, 1,647 rows (785 negative; 862 positive)
- stream: after **2017-11-23**, 11,256 rows (5,641 negative; 5,615 positive)

## Chunking decision
The stream contains 117 observed dates. Daily volume is heterogeneous (median 95, max 370) and there are temporal gaps, including a maximum gap of 145 days. Splitting individual days would manufacture within-day temporal order. Therefore the primary stream uses **complete-day grouped chunks targeting 128 rows**.

Observed primary target-128 behavior:
- 87 chunks
- median 124 rows
- minimum 53; maximum 370
- median 1 observed day per chunk; maximum 3

Sensitivity analyses use targets 64 and 256, also preserving whole days.

## Consequences for claims
1. Public reproducibility claims refer to the exact accessible CSV blobs, not the manuscript's stated 17,149-row composition.
2. The original random train/test files are not used as the primary split.
3. Detection/routing remains label-free; post-prediction adaptation may use delayed labels.
4. Distribution shift is not automatically called concept drift without label-conditioned evidence.
