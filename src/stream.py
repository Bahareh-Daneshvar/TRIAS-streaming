from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    reference: pd.DataFrame
    calibration: pd.DataFrame
    stream: pd.DataFrame


def chronological_split(df: pd.DataFrame, date_col: str, reference_fraction=0.20, calibration_fraction=0.10) -> TemporalSplit:
    if reference_fraction <= 0 or calibration_fraction <= 0 or reference_fraction + calibration_fraction >= 1:
        raise ValueError("Fractions must be >0 and sum to <1")
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    if x[date_col].isna().any():
        raise ValueError("Unparseable dates remain; audit/clean before chronological splitting")
    x = x.sort_values(date_col, kind="stable").reset_index(drop=True)
    n = len(x)
    i = max(1, int(round(n * reference_fraction)))
    j = max(i + 1, int(round(n * (reference_fraction + calibration_fraction))))
    if j >= n:
        raise ValueError("Dataset too small for requested split")
    return TemporalSplit(x.iloc[:i].copy(), x.iloc[i:j].copy(), x.iloc[j:].copy())


def iter_chunks(df: pd.DataFrame, chunk_size: int):
    if chunk_size < 2:
        raise ValueError("chunk_size must be >=2")
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size].copy()
        if len(chunk):
            yield chunk
