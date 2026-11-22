# 有 pyarrow 但 pandas 把 date32 欄讀成 Python date 物件（object）時使用：讀 parquet 後把日期欄轉成 datetime64，
# 行為與 pqshim 一致。用法：PYTHONPATH=lab/tools/pqdates python lab/core/run.py <編號>
import datetime as _dt
import pandas as _pd
_orig = _pd.read_parquet


def _read(*a, **k):
    df = _orig(*a, **k)
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna()
            if len(s) and isinstance(s.iloc[0], _dt.date):
                df[c] = _pd.to_datetime(df[c])
    return df


_pd.read_parquet = _read
