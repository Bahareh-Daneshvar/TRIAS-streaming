from __future__ import annotations
import re
import pandas as pd

ALIASES = {
    "date": ["日期", "date", "Date", "datetime", "timestamp"],
    "label": ["正负面", "正/负面", "label", "Label", "sentiment", "target"],
    "title": ["标题", "title", "headline"],
    "body": ["正文", "正文 ", "body", "text", "content"],
}


def _find(df, names, required=True):
    for c in names:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find any of columns {names}; got {list(df.columns)}")
    return None




def parse_dates(series: pd.Series) -> pd.Series:
    """Parse the public Data 1 Chinese date format plus ordinary ISO-like dates."""
    raw = series.astype(str).str.strip()
    chinese = raw.str.replace("年", "-", regex=False).str.replace("月", "-", regex=False).str.replace("日", "", regex=False)
    return pd.to_datetime(chinese, errors="coerce")

def normalize_label(v):
    s = str(v).strip().lower()
    positive = {"1", "positive", "pos", "正面", "正", "利好", "好"}
    negative = {"0", "-1", "negative", "neg", "负面", "負面", "负", "利空", "坏", "壞"}
    if s in positive:
        return 1
    if s in negative:
        return 0
    raise ValueError(f"Unknown sentiment label: {v!r}; inspect dataset before extending mapping")


def normalize_text(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_data1(df: pd.DataFrame) -> pd.DataFrame:
    dcol = _find(df, ALIASES["date"])
    lcol = _find(df, ALIASES["label"])
    tcol = _find(df, ALIASES["title"])
    bcol = _find(df, ALIASES["body"], required=False)

    out = pd.DataFrame()
    out["date"] = parse_dates(df[dcol])
    if out["date"].isna().any():
        raise ValueError(f"Unparseable dates: {int(out['date'].isna().sum())}")
    out["label"] = df[lcol].map(normalize_label)
    title = df[tcol].map(normalize_text)
    body = pd.Series([""] * len(df), index=df.index) if bcol is None else df[bcol].map(normalize_text)
    out["title"] = title
    out["body"] = body
    out["text"] = (title + " [SEP] " + body).str.strip()
    out["source_index"] = df.index
    return out.sort_values("date", kind="stable").reset_index(drop=True)
