# Data Acquisition Status v0.2

## Dataset 1 — cited by the source study
Repository: `wwwxmu/Dataset-of-financial-news-sentiment-classification`

Public repository metadata verified:
- 17,149 financial-news records reported
- fields reported: date, company, code, sentiment, title, body
- 12,514 positive / 4,635 negative reported
- files: `news_seed.xlsx`, `train_data.csv`, `test_data.csv`

### Current blocker in this sandbox
The repository page is readable, but the runtime cannot resolve/download `raw.githubusercontent.com`. Therefore no row-level audit result is claimed yet.

### Immediate action when GitHub connector or local files are available
1. Fetch exact repository files without modification.
2. Store source hashes.
3. Run `src/data_audit.py`.
4. Check whether `train_data.csv` and `test_data.csv` overlap by title/body/date/company.
5. Parse and sort dates.
6. Quantify label distribution by month/year.
7. Define reference/calibration/stream periods only after seeing temporal density.

## Dataset 2 — external validation shortlist
### Preferred first candidate: FNSPID
Strength: very large, explicitly time-series financial-news dataset with long temporal coverage.
Risk: sentiment labels/score provenance must be audited before using it as supervised ground truth.

### Secondary candidate: Twitter Financial News Sentiment
Strength: human/annotated three-class financial sentiment benchmark.
Risk: public version exposes text and labels but no obvious timestamp field, so it is unsuitable as the main chronological external stream unless original timestamps can be recovered defensibly.

## Decision rule
Do not select Dataset 2 merely because it is large. Temporal order and label provenance are mandatory for the primary external streaming claim.
