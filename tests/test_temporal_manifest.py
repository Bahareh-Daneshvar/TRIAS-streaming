import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from temporal_manifest import chronological_split_by_day, make_manifest


def test_grouped_split_never_splits_a_day():
    rows = []
    for d in pd.date_range("2020-01-01", periods=30, freq="D"):
        for i in range(1 + (d.day % 4)):
            rows.append({"date": d + pd.Timedelta(hours=i), "label": i % 2, "text": f"{d}-{i}"})
    df = pd.DataFrame(rows)
    split = chronological_split_by_day(df, "date", 0.2, 0.1)
    m = make_manifest(split)
    parts_per_day = m.assign(day=m.date.dt.normalize()).groupby("day").partition.nunique()
    assert int((parts_per_day > 1).sum()) == 0
    assert len(split.reference) > 0 and len(split.calibration) > 0 and len(split.stream) > 0
