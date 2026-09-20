from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class GroupedTemporalSplit:
    reference: pd.DataFrame
    calibration: pd.DataFrame
    stream: pd.DataFrame
    reference_end: pd.Timestamp
    calibration_end: pd.Timestamp


def _choose_boundary_by_rows(day_counts: pd.Series, target_rows: int, min_day_index: int = 0) -> int:
    """Return inclusive day index whose cumulative rows is closest to target.

    Boundaries are chosen only between complete calendar days so one date never
    leaks across reference/calibration/stream partitions.
    """
    if len(day_counts) < 3:
        raise ValueError("Need at least three distinct dates for grouped temporal splitting")
    csum = day_counts.cumsum().to_numpy()
    candidates = list(range(max(0, min_day_index), len(day_counts) - 1))
    if not candidates:
        raise ValueError("No valid temporal boundary candidates")
    return min(candidates, key=lambda i: abs(int(csum[i]) - int(target_rows)))


def chronological_split_by_day(
    df: pd.DataFrame,
    date_col: str = "date",
    reference_fraction: float = 0.20,
    calibration_fraction: float = 0.10,
) -> GroupedTemporalSplit:
    if reference_fraction <= 0 or calibration_fraction <= 0 or reference_fraction + calibration_fraction >= 1:
        raise ValueError("Fractions must be >0 and sum to <1")
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    if x[date_col].isna().any():
        raise ValueError("Unparseable dates remain")
    x = x.sort_values(date_col, kind="stable").reset_index(drop=True)
    day = x[date_col].dt.normalize()
    counts = day.value_counts().sort_index()
    if len(counts) < 10:
        raise ValueError("Too few unique dates for a credible chronological stream")

    n = len(x)
    ref_target = round(n * reference_fraction)
    cal_target_cum = round(n * (reference_fraction + calibration_fraction))
    ref_i = _choose_boundary_by_rows(counts, ref_target, 0)
    cal_i = _choose_boundary_by_rows(counts, cal_target_cum, ref_i + 1)
    if cal_i <= ref_i or cal_i >= len(counts) - 1:
        raise ValueError("Could not construct non-empty grouped temporal partitions")

    dates = counts.index
    ref_end = pd.Timestamp(dates[ref_i])
    cal_end = pd.Timestamp(dates[cal_i])
    ref = x[day <= ref_end].copy()
    cal = x[(day > ref_end) & (day <= cal_end)].copy()
    stream = x[day > cal_end].copy()
    if min(len(ref), len(cal), len(stream)) == 0:
        raise ValueError("Grouped temporal split produced an empty partition")
    return GroupedTemporalSplit(ref, cal, stream, ref_end, cal_end)


def make_manifest(split: GroupedTemporalSplit) -> pd.DataFrame:
    frames = []
    for name, part in (("reference", split.reference), ("calibration", split.calibration), ("stream", split.stream)):
        y = part.copy()
        y["partition"] = name
        frames.append(y)
    return pd.concat(frames, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)


def iter_day_grouped_target_chunks(
    df: pd.DataFrame,
    date_col: str = "date",
    target_rows: int = 128,
):
    """Yield chronological chunks without splitting a calendar day.

    The next complete day is appended only when doing so is at least as close
    to ``target_rows`` as closing the current chunk. A single high-volume day
    may therefore exceed the target, which is intentional: calendar-day
    integrity takes precedence over an exact row count.
    """
    if target_rows < 2:
        raise ValueError("target_rows must be >=2")
    if df.empty:
        return
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    if x[date_col].isna().any():
        raise ValueError("Unparseable dates remain")
    x = x.sort_values(date_col, kind="stable").reset_index(drop=True)
    x["__day"] = x[date_col].dt.normalize()
    groups = [(day, g.drop(columns="__day").copy()) for day, g in x.groupby("__day", sort=True)]

    current = []
    current_n = 0
    for _, day_df in groups:
        n = len(day_df)
        if current:
            before = abs(target_rows - current_n)
            after = abs(target_rows - (current_n + n))
            if current_n >= target_rows or (current_n < target_rows < current_n + n and before <= after):
                yield pd.concat(current, ignore_index=True)
                current = []
                current_n = 0
        current.append(day_df)
        current_n += n
    if current:
        yield pd.concat(current, ignore_index=True)
