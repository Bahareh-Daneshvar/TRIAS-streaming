from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

DATE_CANDIDATES = ["日期", "date", "Date", "datetime", "timestamp"]
LABEL_CANDIDATES = ["正负面", "正/负面", "label", "Label", "sentiment", "target"]
TITLE_CANDIDATES = ["标题", "title", "headline"]
BODY_CANDIDATES = ["正文", "正文 ", "body", "text", "content"]




def _parse_dates(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    chinese = raw.str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False)
    return pd.to_datetime(chinese, errors="coerce")

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_text(v):
    if pd.isna(v):
        return ""
    s = re.sub(r"\s+", " ", str(v)).strip().lower()
    return s


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    errors = []
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            df.attrs["encoding"] = enc
            return df
        except Exception as e:
            errors.append((enc, str(e)))
    raise RuntimeError(f"Could not read {path}. Attempts: {errors}")


def audit(paths: list[Path]) -> dict:
    frames = []
    per_file = {}
    for p in paths:
        df = load_any(p)
        enc = df.attrs.get("encoding", "excel/binary")
        df["__source_file"] = p.name
        frames.append(df)
        per_file[p.name] = {
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "sha256": sha256(p),
            "bytes": int(p.stat().st_size),
            "detected_encoding": enc,
        }
    data = pd.concat(frames, ignore_index=True, sort=False)

    date_col = _find_col(data, DATE_CANDIDATES)
    label_col = _find_col(data, LABEL_CANDIDATES)
    title_col = _find_col(data, TITLE_CANDIDATES)
    body_col = _find_col(data, BODY_CANDIDATES)

    clean_cols = [c for c in data.columns if c != "__source_file"]
    report = {
        "files": per_file,
        "combined_rows": int(len(data)),
        "combined_columns": list(map(str, data.columns)),
        "detected": {"date": date_col, "label": label_col, "title": title_col, "body": body_col},
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "exact_duplicate_rows": int(data[clean_cols].duplicated().sum()),
    }

    if label_col:
        vc = data[label_col].astype(str).value_counts(dropna=False)
        report["label_counts"] = {str(k): int(v) for k, v in vc.items()}

    norm_title = data[title_col].map(_norm_text) if title_col else None
    norm_body = data[body_col].map(_norm_text) if body_col else None
    if title_col:
        report["duplicate_titles_normalized"] = int(norm_title.duplicated().sum())
    if body_col:
        nonempty_body = norm_body[norm_body != ""]
        report["duplicate_bodies_normalized_nonempty"] = int(nonempty_body.duplicated().sum())
    if title_col or body_col:
        t = norm_title if title_col else pd.Series([""] * len(data))
        b = norm_body if body_col else pd.Series([""] * len(data))
        fingerprint = t + "\n" + b
        report["duplicate_title_body_fingerprints"] = int(fingerprint.duplicated().sum())
        tmp = pd.DataFrame({"fp": fingerprint, "src": data["__source_file"]})
        cross = tmp.groupby("fp")["src"].nunique()
        report["fingerprints_spanning_multiple_files"] = int((cross > 1).sum())

    if date_col:
        parsed = _parse_dates(data[date_col])
        report["date_parse_failures"] = int(parsed.isna().sum())
        valid = parsed.dropna()
        if len(valid):
            report["date_min"] = str(valid.min())
            report["date_max"] = str(valid.max())
            report["unique_dates"] = int(valid.nunique())
            month = valid.dt.to_period("M").value_counts().sort_index()
            year = valid.dt.to_period("Y").value_counts().sort_index()
            report["monthly_counts"] = {str(k): int(v) for k, v in month.items()}
            report["yearly_counts"] = {str(k): int(v) for k, v in year.items()}
            if title_col:
                tmp = pd.DataFrame({"d": parsed, "t": norm_title}).dropna()
                span = tmp.groupby("t")["d"].nunique()
                report["normalized_titles_on_multiple_dates"] = int((span > 1).sum())
            if title_col or body_col:
                tmp = pd.DataFrame({"d": parsed, "fp": fingerprint, "src": data["__source_file"]}).dropna()
                date_span = tmp.groupby("fp")["d"].nunique()
                report["fingerprints_on_multiple_dates"] = int((date_span > 1).sum())

    # Direct train/test contamination indicator when both names are present.
    source_names = set(data["__source_file"].astype(str))
    if {"train_data.csv", "test_data.csv"}.issubset(source_names) and (title_col or body_col):
        tr = set(fingerprint[data["__source_file"] == "train_data.csv"])
        te = set(fingerprint[data["__source_file"] == "test_data.csv"])
        report["train_test_fingerprint_overlap"] = int(len(tr & te))

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/data_audit.json"))
    args = ap.parse_args()
    report = audit(args.paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
