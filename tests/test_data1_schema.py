import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from data1 import normalize_data1


def test_actual_public_header_variant_is_supported():
    df = pd.DataFrame({
        "日期": ["2019年2月14日", "2018年04月03日"],
        "公司": ["A", "B"],
        "代码": ["1", "2"],
        "正负面": [0, 1],
        "标题": ["坏消息", "好消息"],
        "正文": ["正文一", "正文二"],
    })
    out = normalize_data1(df)
    assert list(out["label"]) == [1, 0]  # sorted chronologically
    assert out["date"].notna().all()
