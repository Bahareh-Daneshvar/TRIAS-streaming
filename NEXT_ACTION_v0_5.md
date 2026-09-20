# TRIAS next action — v0.5

## What is blocked?
Only the row-level Phase-1 run is blocked by missing public Data-1 bytes in this runtime.

## Minimum user input
Either:
1. connect the GitHub plugin so the repository can be read/written directly; or
2. upload `train_data.csv` and `test_data.csv` from the public dataset repository.

`news_seed.xlsx` is optional for the primary journal experiment.

## What will run immediately after the two CSVs are available?
1. cryptographic hashes and encoding audit;
2. exact/near duplicate and original train/test overlap audit;
3. date parsing and temporal-density audit;
4. combination of the historical train/test files for the primary chronological protocol;
5. day-grouped reference/calibration/stream manifest, with no calendar day allowed to cross a boundary;
6. decision on chunk policy from observed temporal density;
7. initial fusion host baseline training;
8. B0–B7 prequential pilot and go/no-go on signature routing.

## Why the original split is not used for the primary stream test
The source paper's static train/test organization is useful for reproduction, but the primary TRIAS claim concerns non-stationary chronological streams. The primary protocol therefore recombines the public files and re-partitions only by time after contamination auditing.
