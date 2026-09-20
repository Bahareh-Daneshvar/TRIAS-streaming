# Data 1 acquisition and audit status v0.3

## Provenance verified
The source paper identifies an open GitHub dataset from `wwwxmu/Dataset-of-financial-news-sentiment-classification`.
The repository README reports 17,149 records with six fields: date, company, code, positive/negative label, title, and body; 12,514 positive and 4,635 negative records.
Repository files shown publicly: `news_seed.xlsx`, `train_data.csv`, `test_data.csv`.

## Runtime acquisition status
The current compute container has no direct outbound DNS/network access, and raw GitHub file download through the available file downloader failed. Therefore **no row-level audit result is claimed yet**.

The GitHub repository page itself is accessible through web retrieval, so provenance and advertised schema are verified, but this is not a substitute for reading all rows.

## Exact next action once GitHub connector or raw files are available
1. acquire train_data.csv, test_data.csv, news_seed.xlsx;
2. record SHA-256 for each file;
3. identify exact encoding;
4. concatenate train/test only after adding source-file provenance;
5. parse dates and report failures;
6. quantify exact-row, title-only, body-only, and title+body duplicates;
7. check duplicate texts spanning different dates and split boundaries;
8. inspect class balance by month/quarter/year;
9. verify whether repository train/test assignment is temporal or random;
10. construct a deduplicated chronological stream;
11. choose reference/calibration windows from temporal density, not arbitrary row fractions;
12. freeze the final stream manifest before model experiments.

## Important scientific constraint
Do not use `news_seed.xlsx` as an additional independent test set without checking overlap, because the repository describes it as the seed data from which the expanded dataset was constructed.
