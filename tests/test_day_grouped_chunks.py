import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from temporal_manifest import iter_day_grouped_target_chunks


def test_day_grouped_chunks_never_split_calendar_day():
    rows=[]
    for d,n in [("2020-01-01",80),("2020-01-02",70),("2020-01-03",200),("2020-01-04",30)]:
        for i in range(n):
            rows.append({"date":pd.Timestamp(d)+pd.Timedelta(minutes=i),"x":i})
    df=pd.DataFrame(rows)
    chunks=list(iter_day_grouped_target_chunks(df,target_rows=128))
    seen={}
    for ci,c in enumerate(chunks):
        for day in c.date.dt.normalize().unique():
            assert day not in seen
            seen[day]=ci
    assert sum(map(len,chunks)) == len(df)
