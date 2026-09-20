from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from data_audit import audit, load_any
from data1 import normalize_data1
from temporal_manifest import chronological_split_by_day, make_manifest


def main():
    ap = argparse.ArgumentParser(description="TRIAS Phase-1 dataset audit and chronology preparation")
    ap.add_argument("--train", type=Path, default=Path("data/raw/train_data.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/raw/test_data.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("outputs/phase1"))
    ap.add_argument("--reference-fraction", type=float, default=0.20)
    ap.add_argument("--calibration-fraction", type=float, default=0.10)
    args = ap.parse_args()

    missing = [str(p) for p in (args.train, args.test) if not p.exists()]
    if missing:
        raise SystemExit("Missing required data files: " + ", ".join(missing))

    args.outdir.mkdir(parents=True, exist_ok=True)
    report = audit([args.train, args.test])
    (args.outdir / "data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # The repository's historical train/test split is ignored for the primary
    # stream protocol. We combine both public files, normalize once, then sort
    # chronologically. Source identity is preserved for contamination analysis.
    pieces = []
    for p in (args.train, args.test):
        raw = load_any(p)
        norm = normalize_data1(raw)
        norm["source_file"] = p.name
        pieces.append(norm)
    data = pd.concat(pieces, ignore_index=True)
    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    split = chronological_split_by_day(
        data,
        date_col="date",
        reference_fraction=args.reference_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    manifest = make_manifest(split)
    manifest.to_csv(args.outdir / "chronological_manifest.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(manifest)),
        "date_min": str(manifest["date"].min()),
        "date_max": str(manifest["date"].max()),
        "unique_days": int(manifest["date"].dt.normalize().nunique()),
        "reference_rows": int(len(split.reference)),
        "calibration_rows": int(len(split.calibration)),
        "stream_rows": int(len(split.stream)),
        "reference_end_day": str(split.reference_end.date()),
        "calibration_end_day": str(split.calibration_end.date()),
        "same_day_cross_partition_violations": 0,
        "label_counts_by_partition": {
            k: {str(lbl): int(n) for lbl, n in g["label"].value_counts().sort_index().items()}
            for k, g in manifest.groupby("partition", sort=False)
        },
        "source_counts_by_partition": {
            k: {str(src): int(n) for src, n in g["source_file"].value_counts().items()}
            for k, g in manifest.groupby("partition", sort=False)
        },
    }
    # Defensive verification: no calendar date may belong to >1 partition.
    day_parts = manifest.assign(day=manifest["date"].dt.normalize()).groupby("day")["partition"].nunique()
    summary["same_day_cross_partition_violations"] = int((day_parts > 1).sum())
    if summary["same_day_cross_partition_violations"]:
        raise RuntimeError("Temporal leakage: at least one calendar day spans multiple partitions")

    (args.outdir / "phase1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
